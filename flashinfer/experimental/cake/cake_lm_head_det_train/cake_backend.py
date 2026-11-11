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
(flashinfer-ai/flashinfer#5680).

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
* Entry (a) ``cake_lm_head_det_train_loss``: the forward produces ``dX_acc`` (FP32
  ``[T, H]``) and ``dW_acc`` (FP32 ``[V, H]``) directly per chunk (three GEMMs,
  no logits recomputation); the backward applies the incoming scalar
  gradient ``g`` and performs the single cast ``dX = bf16(g * dX_acc)``,
  ``dW = cast(g * dW_acc)`` -- never mutating the saved accumulators, so a
  retained graph may run the backward repeatedly.  With the gradient
  placement ``"auto"`` (the default, :func:`set_gradient_placement`) a call no
  backward can follow forms no accumulator, and inside a non-reentrant
  ``torch.utils.checkpoint`` the recomputation forms them instead of the
  checkpointed forward (the same bits).  Entry (b)
  ``cake_lm_head_det_train_logprob``: the forward saves the FP32 row statistic
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
* Fused dX finalize (``dx_finalize``, default on): the K-slice slabs of a sliced dX GEMM are added into ``dX_acc`` by one
  fixed-order kernel (``slab_sum``) instead of the host's chain of in-place
  ``add_`` launches, and a compacted call's dX output boundary writes the
  ``[T, H]`` BF16 ``dX`` in one pass (``scale_cast_scatter_bf16``: ``bf16(g *
  dX_acc[compact row])`` on the valid rows, exact zeros elsewhere, addressed by
  the inclusive int32 scan of the valid-row mask) instead of the flat cast, the
  zero fill and the ``index_copy_``.  The same FP32 operations in the same
  order: ``dX`` is bitwise the previous path's (``dx_finalize=False`` restores
  that path).
* dW side stream (``dw_stream``: ``None`` / ``"auto"`` (default) = calls of
  three or more chunks, ``True`` = every multi-chunk call, ``False`` = never):
  each chunk's weight-gradient accumulate GEMM is launched on a
  per-device side stream forked after the chunk's ``row_grad`` (``dz`` ready)
  and joined before the next chunk reuses the chunk buffer and before the call
  returns, so its CTAs fill the tail wave of the chunk's dX GEMM instead of
  queueing behind it.  Kernels, buffers, launch order per kernel and numerics
  are unchanged (the dX GEMM writes the dX rows / slabs, the accumulate reads
  the same ``dz`` and read-modify-writes ``dW_acc``): ``loss`` / ``logp`` /
  ``dX`` / ``dW`` are bitwise the single-stream path's.  Two-chunk calls pay
  the join without an overlap gain and a one-chunk call defers its only
  accumulate to the backward, hence the ``auto`` rule (:func:`dw_side_stream`).
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
  :func:`forward_loss` and the loss autograd entry of at least :func:`zero_weight_min_rows` rows): a valid row whose
  ``loss_weights`` entry is exactly zero (``+0`` or ``-0``; known before the forward, like a ``-100`` label)
  contributes exactly zero loss and zero gradient, so the chunk loop runs over the gradient rows
  ``G = (labels >= 0) & (loss_weights != 0)`` only (a NaN weight stays in ``G``).  Both row counts come back in the
  call's one count readback (host path: one 8-byte copy; hidden valid-row count: the same pinned copy, 8 bytes).  The
  zero-weight rows ``Z = (labels >= 0) & (loss_weights == 0)`` get their ``logp`` from a logp-only pass of the
  log-probability forward kernels (``gemm_logits_lbl`` + ``row_finalize``, chunk-bounded buffers) when ``logp`` is
  returned, scattered into the same ``[T]`` output; their ``dX`` rows are exact ``+0``.  The loss
  (:func:`set_zero_weight_loss_mode`): ``"exact"`` (the default) keeps every chunk's per-row loss terms, places them
  at the rows' positions among the valid rows (a zero-weight row's slot holds its weight, the ``+0`` / ``-0`` term
  the plain path forms) and reduces them with the plain path's chunk boundaries and ``loss_reduce`` launches, so
  ``loss`` is bitwise the call without the split; ``"regroup"`` reduces the gradient rows' terms over the split's
  own chunks (the same FP64 sum over different chunk boundaries).  ``logp`` / ``lse`` of every row are bitwise the
  plain path's; ``dX`` rows of ``G`` too whenever their chunk runs the same dX form and K-slice count; ``dW`` groups
  its K reduction over the split's chunk rows (within the accuracy budget; run to run bitwise).  A zero-weight row is
  treated like an ignored row for the gradients: a non-finite value in its ``X`` row, its logits or its
  ``infer_logp`` does not reach ``dX`` / ``dW`` (the plain path gives NaN there through ``0 * inf``; its loss term is
  ``+-0`` either way).  Not applied (the plain path runs): ``compact_rows=False``, ``objective="ce"``, the
  log-probability entry, the prepared runner (:func:`prepare_lm_head_det_loss`) and a caller-given
  ``num_valid_rows`` (the opt-in module; its ``"budget"`` form splits the rows as here).
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
* The mode: fixed sequential chunk order, no atomics -- bitwise reproducible
  ``loss``, ``logp``, ``dX`` and ``dW`` across runs.  The det name is the
  mode, so no function or autograd ``Function`` of this package takes a
  ``deterministic`` argument (passing one is a ``TypeError``), and torch's
  deterministic mode (``torch.use_deterministic_algorithms``) changes nothing.

Every allocation happens in :func:`prepare_lm_head_det_loss`; the returned runner
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
slots the current tensors, freshly allocated outputs and per-call scratch into
the remembered argument plans and launches.  A call without rows (``T == 0``)
returns loss 0, an empty ``logp`` and zero gradients without binding or
launching anything.

Every switch of this module (``compact_rows``, ``fuse_dw_cast``,
``dx_finalize``, ``dw_stream``, ``hidden_count``, ``gemm_tuning``,
``logprob_dz``, ``logprob_lbl``) is a keyword argument of the entry points
that it affects, ``None`` selecting the documented default (the
``*_default()`` functions); the binding cache is configured through
:meth:`BindingCache.configure`, the loss entry's gradient placement through
:func:`set_gradient_placement` and the log-probability backward's
zero-gradient row skip through :func:`set_zero_dlogp_min_rows` /
:func:`set_zero_dlogp_temperature_dz_counted` and the loss entry's zero-weight
split through :func:`set_zero_weight_min_rows` / :func:`set_zero_weight_loss_mode`.
This module reads no environment variable.
FlashInfer's public wrapper (``flashinfer.cake_lm_head_det_train``) owns the A/B
environment variables of the first six and passes the resolved values down;
it leaves the log-probability arms at their defaults.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import inspect
import math
import re
import struct
import threading
import types
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, ClassVar, NamedTuple, Optional, Union, cast

import torch
import torch.utils.checkpoint as _torch_checkpoint
import torch.utils._python_dispatch as _torch_dispatch

from .cake_jit import (
    BASE_STAGES,
    GEMM_STAGES,
    MODULES,
    STAGES,
    STAGES_WITH_TEMPERATURE,
    TEMPERATURE_GEMM_STAGES,
    load_cake_lm_head_det_loss_module,
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
SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}

# Host binding profile: the keyword set of the generated kernels (see
# :func:`stage_values`: the host provides, per stage, the kernel's own argument
# names; a kernel's argument plan selects from them).  A record with another
# ``abi`` (a plumbing placeholder program) is not served by this backend.
ABI_CONTRACT = "lm_head_loss_v1"
# The hidden valid-row count serves calls of this many chunks or more (``ceil(T / chunk)``): its fixed host cost per call (the
# plan on the count after the readback, the launches it orders) is hidden behind chunk 0's logits GEMM only from there; one- and
# two-chunk calls measured 5-16 % slower with it than on the host-count path (same kernels, bitwise the same outputs).
HIDDEN_COUNT_MIN_CHUNKS = 3
# Calls of at most this many rows form the valid-row count and index of the host-count path on the host
# (:func:`valid_rows`): the bool mask ``labels >= 0`` comes back in one copy of ``T`` bytes -- the call's one
# synchronization -- and the index goes back in one copy, instead of ``nonzero``'s count, scan and select kernels and
# their API calls.  Above it the device forms the index.  The same index and mask either way.  With some rows ignored,
# the index copy back makes :func:`valid_rows` alone a few microseconds slower than the device path from about 128 rows
# up (2-6 us measured at 512-1024 rows).
HOST_COUNT_MAX_ROWS = 1024
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
# ``loss_acc`` / ``loss_out`` cells and, for the casts, ``acc`` / ``g`` / ``out``.
# Scalars: the GEMM launch scalars ``GEMM_SCALARS`` (``M`` / ``m_tiles`` /
# ``k_iters`` / ``first_chunk`` / ``ws_slab`` and the geometry ``ldc``, the K
# slicing ``k_slices`` / ``k_slice_iters``, the raster ``group_m`` / ``group_n``:
# :func:`gemm_scalars`), ``rows_c`` / ``row0`` / ``V`` / ``num_tiles`` / ``mode``
# / ``loss_div`` / ``last_chunk`` / ``d_off`` of the row kernels, ``num_vecs`` of
# the casts, and ``T`` / ``H`` / ``C`` for grid rules.
STAGE_TENSORS = {
    **{stage: ("A", "B", "C", "STATS_OUT", "WS") for stage in GEMM_STAGES},
    # the log-probability arms of the logits GEMM also read the per-token rows (``d_in`` = the kernel's ``dlogp``)
    "gemm_logits_dz": ("A", "B", "C", "STATS_OUT", "WS", "labels", "lse", "d_in"),
    "gemm_logits_lbl": ("A", "B", "C", "STATS_OUT", "WS", "labels"),
    # the zero-gradient row skip's counted forms (``_dcnt``, :data:`DCNT_BASES`) also read the chunk's count cell
    **{
        stage: ("A", "B", "C", "STATS_OUT", "WS", "CNT")
        for stage in GEMM_STAGES
        if stage.endswith("_dcnt")
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
    "slab_sum": ("dx", "ws"),
    "scale_cast_scatter_bf16": ("acc", "g", "scan", "idx_lo", "out"),
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
# ``ws_slab`` (elements between the K-slice slabs of ``WS``), ``ldc`` (the output's leading dimension: V of the logits
# GEMM, H of the dX and weight-gradient GEMMs), ``k_slices`` / ``k_slice_iters`` (K-slice work items per tile and the K
# steps of one), ``group_m`` (raster height in row tiles) and ``group_n`` (raster block width in column ITEMS: the whole
# row of items for the 1-D raster).  The generated kernels take the call geometry through them, so one record serves
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
# K-sliced form of the dX GEMM: ``gemm_dx_s`` runs ``k_slices`` K-slice work items per output tile (a
# persistent grid with ``k_slices`` x more items fills the last wave; the count and the slice length
# ``k_slice_iters`` are launch scalars, 2 .. ``DX_K_SLICES_MAX``); slice 0 writes ``dX_acc``, slice ``s >= 1``
# writes slab ``s - 1`` of the FP32 workspace ``WS [k_slices - 1, rows, H]`` (``ws_slab`` = elements between
# slabs) and the host adds the slabs into ``dX_acc`` in fixed slab order (one RN add per element per
# slab, no atomics).  The slice count of a chunk is chosen from its row count and the SM count
# (:func:`recommended_k_slices`) and fitted to the K steps so that every slice holds one (:func:`dx_slices_fit`).
# ``WS`` of the other GEMMs is an unused 16-float dummy (``ws_slab`` 0).
DX_SLICED_STAGE = "gemm_dx_s"
# Fused weight-gradient output: the LAST chunk's ``dz_c^T @ X_c`` GEMM run in the backward (where the upstream scalar
# gradient ``g`` is known) with the scale and the output cast fused into its epilogue, ``C = cast(g * (WS + tile))`` --
# or ``cast(g * tile)`` for a one-chunk plan, which then has no FP32 ``[V, H]`` accumulator at all.  ``WS`` is the FP32
# sum of the chunks before it (read, never written), ``STATS_OUT`` the ``[1]`` FP32 scale cell.  The same FP32
# operations in the same order as ``gemm_dw_acc`` followed by ``scale_cast_*``, so the weight gradient is bitwise
# identical; the flat ``scale_cast`` pass over ``dW`` disappears.  Selected by ``fuse_dw_cast`` (default on).
DW_CAST_STAGES = ("gemm_dw_cast_bf16", "gemm_dw_cast_f32")


def _dw_cast_f32(stage: str) -> bool:
    """Whether the stage that writes the final ``dW`` (:attr:`Plan.dw_cast_stage`) writes it in FP32: the flat FP32
    cast or the fused FP32 cast in every form (the 2-D blocked ``_gn`` raster included)."""
    return stage == "scale_cast_f32" or stage.startswith("gemm_dw_cast_f32")


# TMA-epilogue form of the deferred BF16 dW cast (``gemm_dw_cast_bf16_tma``): the old FP32 sum is read and the
# BF16 tile is stored as TMA sub-tiles.  It runs the cast of the LAST chunk of a call of >= 2 chunks whose last
# chunk has 1 .. DW_CAST_TMA_MAX_ROWS[arch] rows (:func:`dw_cast_tma_rule`); the per-element (old + acc) * g
# and its round-to-nearest order are the shipped form's, so dW is bitwise the same.  A one-chunk call has no old
# sum to read and keeps the shipped form.
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


# Fused dX finalize (``dx_finalize``, default on): ``slab_sum`` adds the FP32 K-slice slabs of a sliced dX GEMM into
# the chunk's rows of ``dX_acc`` in ascending slab order in one launch (one RN add per element per slab -- the
# evaluation order of the host's ``add_`` chain, ``dX_acc`` read and written once instead of once per slab), and
# ``scale_cast_scatter_bf16`` writes a compacted call's ``[T, H]`` BF16 ``dX`` in one pass: ``bf16(g * dX_acc[scan[r]
# - 1])`` on a valid output row ``r`` (``scan`` = the inclusive int32 count of valid rows, so ``idx[scan[r] - 1] == r``
# exactly on the valid rows), exact zeros elsewhere -- the flat ``scale_cast``, the zero fill and the ``index_copy_``
# of the compact rows in one kernel that writes every output element once.  Bitwise the same ``dX``.
DX_FINALIZE_STAGES = ("slab_sum", "scale_cast_scatter_bf16")
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
K_SLICE_PENALTY = 0.01  # wave-efficiency score penalty per extra slab (the kernels' fitted per-slab cost share)

# --------------------------------------------------------------------------- per-chunk instance rules
# The GEMM instances of a record come in structural FORMS -- one stage name per kernel body the generated program carries
# (:func:`stage_variant`: the device-count logits GEMM ``_mcnt``, the K-sliced ``_s`` / narrow-tile ``_tn256`` / 3-deep-ring
# ``_st3`` dX GEMMs, the 2-D blocked ``_gn`` weight-gradient GEMMs) -- and take the call geometry and the rules' numeric
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
#   floor (the badly quantized chunk tails); the slice count (the ``_s`` form with ``k_slices`` 2 .. DX_K_SLICES_MAX)
#   comes from :func:`recommended_k_slices` as before.  Bitwise: the same slice boundaries and per-element K order on
#   both tiles.
# * dX operand ring (``_st3``): below RASTER_RULE_MIN_HIDDEN a chunk of >= DW_LONG_CHUNK_MIN_ROWS rows runs the 512-wide dX
#   tile with the 3-deep operand ring (:func:`dx_stages_variant`, DX_LONG_CHUNK_STAGES per architecture); the 256-wide
#   fallback keeps its depth.  Bitwise: the ring depth only changes when a stage is refilled.
# * dW 2-D blocked raster (``_gn`` with ``group_n``): below RASTER_RULE_MIN_HIDDEN a chunk of DW_BLOCK_MIN_ROWS ..
#   DW_BLOCK_MAX_ROWS rows runs the weight-gradient accumulate and the fp32 fused cast with column blocks of
#   DW_BLOCK_GROUPS[arch] tiles (:func:`dw_block_variant`; the bf16 cast keeps the 1-D raster).  Bitwise: a permutation
#   of the same tiles.
# ``gemm_tuning`` (a keyword of :func:`prepare_lm_head_det_loss` and of the eager / autograd entry points; ``None`` = no
# explicit knob) pins knobs explicitly per GEMM -- ``{"logits": {"group_m": 16}, "dx": {"tile_n": 256, "k_slices": 2},
# "dw": {"group_m": 32, "group_n": 0}}``; a knob given as ``None`` pins the default -- and wins over the rule for the
# knobs it names.  A form the record does not register fails closed (``NotImplementedError``).
RASTER_RULE_MIN_HIDDEN = 7168  # first H whose [rows_c, H] BF16 operand panel no longer stays L2-resident at rows_c >= 2049
RASTER_RULE_MIN_ROWS = 2049
RASTER_WIDE_GROUPS = {
    "sm_100a": (16, 32),
    "sm_103a": (16, 32),
    "sm_107a": (32, 32),
}  # (logits group_m, weight-gradient group_m) of the wide geometries, per architecture (sm_107a: the logits slot keeps
#    the record's group_m 32)
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
}  # logits group_m of the GLM-class chunks of >= LOGITS_LONG_RASTER_MIN_ROWS rows below RASTER_RULE_MIN_HIDDEN, per architecture
#    (sm_107a is not listed: its chunks keep the record's logits group_m 32)
LOGITS_LONG_RASTER_MIN_ROWS = (
    3841  # 30 row tiles of 128 + 1: the first chunk size with 31 row tiles
)
DX_LONG_CHUNK_STAGES = {
    "sm_100a": 3,
    "sm_103a": 3,
}  # operand-ring depth of the 512-wide dX tile at the long chunks (>= DW_LONG_CHUNK_MIN_ROWS rows) below RASTER_RULE_MIN_HIDDEN
#    (sm_107a is not listed: its long chunks keep the table ring depth DX_WIDE_TILE_STAGES)
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
DX_K_SLICES_MAX = 4
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
)  # architectures whose long-chunk ring (``_st3``) also takes the split (on sm_100a it ran slower at the power limit; on
#    sm_107a only an explicit ``stages`` pin selects the ring)
DX_SPLIT_WHOLE_ARCHES: tuple[str, ...] = (
    "sm_103a",
)  # whole-tile chunks below RASTER_RULE_MIN_HIDDEN split in every call
DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES: tuple[str, ...] = (
    "sm_100a",
    "sm_107a",
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
    r"(?P<mcnt>_mcnt)?(?P<s>_s)?(?:_tn(?P<tile>\d+))?(?:_st(?P<stages>\d+))?(?P<split>_split)?(?P<swab>_swab)?"
    r"(?P<gn>_gn)?(?P<tma>_tma)?(?P<dcnt>_dcnt)?(?P<tau>_tau)?"
)
STAGE_KNOBS = (
    "count",
    "sliced",
    "tile_n",
    "stages",
    "split",
    "swab",
    "blocked",
    "epi_store",
    "dcnt",
    "temperature",
)
# The zero-gradient row skip of the log-probability backward (``_dcnt``, :func:`zero_dlogp_partition`): the backward
# orders each chunk's rows so that the rows whose upstream gradient ``dlogp`` is nonzero come first and holds their
# number per chunk in an int32 device cell (``CNT``); the ``_dcnt`` form of a GEMM bounds its work by that cell at the
# launch shape of the chunk's full row count (no read-back, one launch sequence, CUDA-graph capturable).  The dz arm
# skips the work items whose pair of row tiles starts at or past the count (their rows keep what they held: never
# read), the weight-gradient accumulate and bf16 fused cast run ``ceil(count / 64)`` of its K steps (one at least on a
# first chunk; none leaves the accumulator as it was, and the cast writes ``bf16(g * WS)``).  The counted forms are
# this package's GEMM template arithmetic with a device row bound: each output element keeps one owner per launch and
# the chunks run in order, so they are deterministic.  The dX GEMM has no counted form here: it computes every row of
# its chunk, as without the skip (the rows past the count read stale ``dz`` and yield ``dX_acc`` rows nothing reads,
# :func:`finalize_dx`).  The bases that have the form (not the accumulate's TMA epilogue; the bf16 fused cast's TMA
# store has it too, :data:`DW_CAST_TMA_DCNT_STAGE`):
DCNT_BASES = (LOGPROB_DZ_STAGE, "gemm_dw_acc", "gemm_dw_cast_bf16")


def stage_variant(
    base: str,
    *,
    count: bool = False,
    sliced: bool = False,
    tile_n: Optional[int] = None,
    stages: Optional[int] = None,
    split: bool = False,
    swab: bool = False,
    blocked: bool = False,
    epi_store: Optional[str] = None,
    dcnt: bool = False,
) -> str:
    """Stage name of one structural form of ``base``: the rule outputs that select a kernel body, in the fixed suffix
    order ``_mcnt``, ``_s``, ``_tn256``, ``_st<stages>``, ``_split``, ``_swab``, ``_gn``, ``_tma``, ``_dcnt``; ``split``: the wide
    dX tile with its last wave's trailing items run as half items (:func:`dx_split_rule`); ``swab``: the wide dX tile
    with swapped operands at the table ring depth (:func:`dx_swab_rule`); a default output (the host-bound row
    count, one slice, the wide tile, the table ring depth, the 1-D raster, the default epilogue) adds nothing, so
    ``stage_variant(base)`` is ``base`` itself.  ``count``: the logits GEMM whose valid-row bound is read from device
    memory (chunk 0 of the hidden valid-row count path); ``sliced``: the dX GEMM run as ``k_slices > 1`` K-slice work
    items per tile; ``blocked``: the weight-gradient accumulate / fp32 fused cast on the 2-D blocked raster
    (``group_n`` set); ``epi_store="tma"``: the weight-gradient accumulate's TMA reduce-add or the bf16 fused cast's
    TMA store (:func:`dw_cast_tma_rule`); ``dcnt``: the form bounded by the device count of the chunk's live rows
    (:data:`DCNT_BASES`, the zero-gradient row skip; the bf16 fused cast's TMA store has one too,
    :data:`DW_CAST_TMA_DCNT_STAGE`).  The slice count, the raster height and the block width themselves are
    launch scalars (:func:`gemm_scalars`), not part of the name."""
    if base not in BASE_STAGES:
        raise ValueError(f"unknown base stage {base!r}")
    name = base
    if count:
        if base not in ("gemm_logits", "gemm_logits_nostats"):
            raise ValueError(
                f"{base}: the device-count form exists for the logits GEMMs only"
            )
        name += "_mcnt"
    if sliced:
        if base != "gemm_dx":
            raise ValueError(
                f"{base}: the K-slice work-item form exists for gemm_dx only"
            )
        name += "_s"
    if tile_n is not None and int(tile_n) != DX_TILE_WIDE:
        if base != "gemm_dx" or int(tile_n) != DX_TILE_NARROW:
            raise ValueError(
                f"{base}: the only non-default dX tile is {DX_TILE_NARROW}, got {tile_n}"
            )
        name += f"_tn{DX_TILE_NARROW}"
    if stages is not None and int(stages) != DX_WIDE_TILE_STAGES:
        if (
            base != "gemm_dx"
            or int(stages) not in set(DX_LONG_CHUNK_STAGES.values())
            or (tile_n is not None and int(tile_n) != DX_TILE_WIDE)
        ):
            raise ValueError(
                f"{base}: the only ring-depth form is the 512-wide dX tile's {sorted(set(DX_LONG_CHUNK_STAGES.values()))}-deep ring, got {stages} (tile {tile_n})"
            )
        name += f"_st{int(stages)}"
    if split or swab:
        if base != "gemm_dx" or (tile_n is not None and int(tile_n) != DX_TILE_WIDE):
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
    if dcnt:
        if base not in DCNT_BASES or (
            name.endswith("_tma") and base != "gemm_dw_cast_bf16"
        ):
            raise ValueError(
                f"{base}: the device-count row bound exists for {DCNT_BASES} only (not the accumulate's TMA-epilogue form)"
            )
        name += "_dcnt"
    return name


def stage_form(base: str, *, temperature: bool = False, **knobs: Any) -> str:
    """Stage name of one structural form of ``base`` over every knob of :data:`STAGE_KNOBS` (the inverse of
    :func:`parse_stage`): :func:`stage_variant` of ``knobs``, then ``temperature``: the temperature form
    (:data:`TEMPERATURE_STAGES`, suffix ``_tau`` last) of the log-probability entry's logits GEMM arms, of the
    device-count logits GEMM with the statistics and of ``row_finalize`` -- of these only the ``dz`` arm in a ``_dcnt``
    form (:data:`LOGPROB_DZ_DCNT_TAU_STAGE`: under the zero-gradient row skip a call with a temperature runs it beside
    the counted dW forms, :func:`make_plan_temperature`).  The default path names its stages with
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
        sliced=bool(m["s"]),
        tile_n=int(m["tile"]) if m["tile"] else None,
        stages=int(m["stages"]) if m["stages"] else None,
        split=bool(m["split"]),
        swab=bool(m["swab"]),
        blocked=bool(m["gn"]),
        epi_store="tma" if m["tma"] else None,
        dcnt=bool(m["dcnt"]),
        temperature=bool(m["tau"]),
    )
    if stage_form(m["base"], **knobs) != stage:
        raise ValueError(f"not a stage name of this backend: {stage!r}")
    return m["base"], tuple(knobs.items())


def parse_stage(stage: str) -> tuple[str, dict[str, Any]]:
    """``(base, {count, sliced, tile_n, stages, split, swab, blocked, epi_store, dcnt, temperature})`` of a stage name (the inverse of
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
    if base == "gemm_dx":
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
    form's inputs, so ``dX`` is bitwise the same on both tiles).  Deterministic in the shapes, the slice count, the SM
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
    that serves ``dx_max`` dX slices (:func:`dx_max_slices`); a program is complete for an entry when it registers them.
    The forms do not depend on the call geometry (``H`` / ``V`` and the rules' numeric outputs are launch scalars), so a
    record serves every admissible geometry with them."""
    st_long = DX_LONG_CHUNK_STAGES.get(arch)
    blocked = arch in DW_BLOCK_GROUPS
    out: set[str] = set()
    for base in bases:
        if base in ("gemm_logits", "gemm_logits_nostats"):
            # the device-count form of the hidden valid-row count path (chunk 0 at the buffer extent min(chunk, T))
            out.add(stage_variant(base, count=True))
        elif base == "gemm_dx":
            for sliced in (False, True) if int(dx_max) > 1 else (False,):
                for tile in (DX_TILE_WIDE, DX_TILE_NARROW):
                    out.add(stage_variant(base, sliced=sliced, tile_n=tile))
                if st_long is not None:
                    out.add(
                        stage_variant(
                            base, sliced=sliced, tile_n=DX_TILE_WIDE, stages=st_long
                        )
                    )
                if arch in DX_SPLIT_ARCHES:
                    for st in (None, st_long):
                        out.add(
                            stage_variant(base, sliced=sliced, stages=st, split=True)
                        )
                if arch in DX_SWAB_ARCHES:
                    out.add(stage_variant(base, sliced=sliced, swab=True))
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
) -> str:
    """Stage name of the dX GEMM run as ``k_slices`` K-slice work items per output tile (the ``_s`` form above one;
    ``tile_n`` = the pair tile's column width, ``None`` / ``DX_TILE_WIDE`` = the wide default; ``stages`` = the
    operand-ring depth, ``None`` = the table depth)."""
    if not 1 <= int(k_slices) <= DX_K_SLICES_MAX:
        raise ValueError(
            f"the dX GEMM runs 1 .. {DX_K_SLICES_MAX} K slices per tile, got {k_slices}"
        )
    return stage_variant(
        "gemm_dx",
        sliced=int(k_slices) > 1,
        tile_n=tile_n,
        stages=stages,
        split=split,
        swab=swab,
    )


def dx_max_slices(stages) -> int:
    """Largest slice count the registered stages serve: ``DX_K_SLICES_MAX`` when the K-sliced form ``DX_SLICED_STAGE``
    is registered (the count is a launch scalar of that one form), else ``1``."""
    return DX_K_SLICES_MAX if DX_SLICED_STAGE in stages else 1


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
    the dX GEMM.  The logits GEMM always stores (``first_chunk`` 0 = no accumulate), the dX GEMM always stores its chunk's
    rows (1); ``first_chunk`` / ``ws_slab`` are taken from the arguments for the weight-gradient / dX GEMM only.  The
    form of ``stage`` must agree with the scalars (the ``_s`` form with ``k_slices > 1``, the ``_gn`` form with a
    ``group_n``).  A pure function of its arguments, remembered per argument tuple (bounded; a rejected tuple is not
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
    _base, knobs = parse_stage(stage)
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
    elif not 1 <= ks <= DX_K_SLICES_MAX or bool(knobs["sliced"]) != (ks > 1):
        raise ValueError(
            f"{stage}: the K-sliced form runs 2 .. {DX_K_SLICES_MAX} slices and the plain form one, got k_slices={k_slices}"
        )
    if ks > k_iters:
        raise ValueError(f"{stage}: {ks} K slices over {k_iters} K steps")
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


def dx_slices_fit(k_slices: int, k_iters: int) -> int:
    """The slice count that runs ``k_slices`` requested slices over ``k_iters`` K steps with every slice holding at least
    one step (the reduction form's launch contract ``k_slice_iters * (k_slices - 1) < k_iters`` with ``k_slice_iters =
    ceil(k_iters / k_slices)``): ``ceil(k_iters / ceil(k_iters / k_slices))`` -- ``k_slices`` itself unless the
    ceil-sized slices cover ``K`` with fewer of them (e.g. 3 requested over 4 steps run as 2)."""
    ks, k = max(1, int(k_slices)), max(1, int(k_iters))
    ksi = -(-k // ks)
    return -(-k // ksi)


def recommended_k_slices(
    rows_c: int,
    hidden: int,
    num_sms: int,
    max_slices: int,
    geometry: "Geometry",
    resident: Optional[int] = None,
    vocab: Optional[int] = None,
) -> int:
    """Slice count of the dX GEMM for a chunk of ``rows_c`` rows: the ``S`` in ``[1, max_slices]`` with the best
    wave efficiency net of ``K_SLICE_PENALTY`` per extra slab.  Deterministic in the shapes, the SM count and the
    device's co-resident cluster count (:func:`cluster_resident`).  ``vocab`` (the dX GEMM's K = V; every planner
    passes it) fits the count to the K steps (:func:`dx_slices_fit`); ``None`` = the wave rule alone."""
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
    if vocab is None:
        return best
    return dx_slices_fit(best, int(vocab) // int(geometry.k_block))


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


def capped_k_slices(
    k_slices: int, cap: Optional[int], vocab: int, geometry: "Geometry"
) -> int:
    """A chunk's dX slice count under the cap ``cap`` (``None``: unchanged): above it, ``cap`` fitted to the K steps
    (:func:`dx_slices_fit`, never above ``cap``).  The zero-weight split (:func:`forward_loss`) caps its chunk loop at
    the largest slice count of the same call without the split, so its K-slice workspace never outgrows that call's."""
    if cap is None or int(k_slices) <= int(cap):
        return int(k_slices)
    return dx_slices_fit(max(1, int(cap)), int(vocab) // int(geometry.k_block))


def dx_slices_max(
    rows: int,
    *,
    hidden: int,
    vocab: int,
    chunk: int,
    num_sms: int,
    dx_max: int,
    geometry: "Geometry",
    resident: Optional[int],
    explicit_k: Optional[int] = None,
    cap: Optional[int] = None,
) -> int:
    """The largest dX slice count over the chunks of a chunk loop of ``rows`` rows without building the plan:
    :func:`make_plan`'s slice rule (an explicit ``k_slices`` knob ``explicit_k`` wins; ``cap``: its ``dx_slices_cap``);
    1 without K slicing.  :attr:`Plan.dx_ws_slabs` is this minus one."""
    if explicit_k is not None:
        return int(explicit_k)
    if int(dx_max) <= 1 or int(rows) <= 0:
        return 1
    return max(
        capped_k_slices(
            recommended_k_slices(
                rows_c, hidden, num_sms, dx_max, geometry, resident, vocab=vocab
            ),
            cap,
            vocab,
            geometry,
        )
        for _, rows_c in plan_chunks(int(rows), int(chunk))
    )


def _dx_reduce(values: dict[str, Any]) -> None:
    """Add the K-slice slabs into the chunk's rows of ``dX_acc`` in fixed slab order (host side, in place)."""
    out, ws, rows_c, k = (
        values["acc"],
        values["ws"],
        int(values["rows_c"]),
        int(values["k_slices"]),
    )
    for slab in range(k - 1):
        out.add_(ws[slab, :rows_c])


# --------------------------------------------------------------------------- valid-row compaction

# The switches below are keyword arguments of the entry points (``None`` = the documented default, the constant each
# ``*_default()`` returns); this module reads no environment variable.  FlashInfer's public wrapper
# (``flashinfer.cake_lm_head_det_train``) owns the A/B environment variables of these switches and passes the resolved values
# down.


def logprob_dz_default() -> bool:
    """Default of ``logprob_dz`` (``True``; the log-probability backward): each chunk's recompute is one
    ``gemm_logits_dz`` launch whose epilogue writes the row gradient ``dz`` in place of the logits where the record
    registers that arm; ``False`` (or a record without it) recomputes the logits with ``gemm_logits_nostats`` and forms
    ``dz`` with ``row_grad`` (bitwise the same ``dz``, hence the same ``dX`` / ``dW``)."""
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
    """Default of ``dx_finalize`` (``True``): the K-slice slabs of a sliced dX GEMM are added by the ``slab_sum`` kernel
    and a compacted call's ``dX`` output is written by the ``scale_cast_scatter_bf16`` kernel
    (:data:`DX_FINALIZE_STAGES`); ``False`` adds the slabs with in-place ``add_`` launches on the host and casts,
    zero-fills and ``index_copy_``-scatters the compact rows (bitwise the same ``dX``).  Resolved once per forward; the
    backward follows the forward's choice through the saved valid-row mask."""
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
    """The calling thread's per-device dict ``name`` of the calls' host state -- ``"dw_streams"`` (device index ->
    (side stream, fork event), :func:`_dw_stream`), ``"cells"`` / ``"pair_cells"`` (device index -> (pinned int32
    [1] / [2] cell, event), :func:`_hidden_count_cell` / :func:`_hidden_count_pair_cell`), ``"arange"`` (device
    index -> the growing int32 constant 1, 2, ..,
    :func:`_hidden_count_arange`) and ``"arange_streams"`` (device index -> the raw streams the constant was recorded
    on).  None is a tensor of a workspace or a binding.  Calls of different threads on one
    device (each on its own stream) never share a stream, an event, a cell or the constant: one thread's call cannot
    re-record another's fork event, overwrite its count before the read, or reallocate the constant its queued
    kernels read; one thread runs one call at a time."""
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
    chunks: 768 dX work items on 76 CTA pairs = 11 waves at 0.919 fill).  Measured on the same device in one process
    (ratio of the step's GPU span to the single-stream step): four-chunk calls 0.991 / 0.995 cross-entropy / policy
    (SM103) and 0.998 / 0.995 (SM100), eight-chunk 2048-row calls 0.996 / 0.993; two-chunk calls pay the join without an
    overlap gain (0.999 / 1.002) and a one-chunk call defers its only accumulate to the backward -- hence the ``auto``
    rule of :data:`DW_STREAM_MIN_CHUNKS` chunks.  Kernels, buffers, launch order per kernel and numerics are unchanged
    (the dX GEMM writes the dX rows / slabs, the accumulate reads the same ``dz`` and read-modify-writes ``dW_acc``;
    each kernel keeps its own order): only the launch stream differs, so every output is bitwise the single-stream
    path's.  The side stream and its fork event are per thread and per device (:func:`_thread_state`): calls of
    different threads on one device never share them."""
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
    """The calling thread's ``(side stream, fork event)`` of device ``index`` (created on first use; never part of a
    binding or a workspace)."""
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
    """The calling thread's ``(pinned int32 [1] cell, event)`` of device ``index``: the count's copy target and the
    event behind it, reused by the thread's calls (each reads its count before the next call)."""
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
    """The calling thread's ``(pinned int32 [2] cell, event)`` of device ``index``: the hidden count of a zero-weight
    split (:func:`_hidden_count_begin` with ``loss_weights``), whose two counts travel in one copy."""
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
    """The calling thread's int32 constant ``1, 2, ..`` of device ``index`` with at least ``T`` entries (no kernel per
    call; grown by doubling), recorded on the current stream so a replaced constant is not reused before the
    kernels queued on it have read it.  The allocator keeps every recorded stream until the constant is freed, so
    each stream is recorded once per constant (``"arange_streams"``), not once per call.  Thread state, not part of
    a call: formed and recorded outside active torch dispatch modes (see :func:`_device_constants`; the backward's
    recomputation of a checkpoint runs on the autograd engine's thread)."""
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
    # the zero-weight split (:func:`gradient_index`; ``None`` / 0 without it): ``zero_pos`` the inclusive int32 scan of
    # the zero-weight mask (released once :meth:`zero_index` formed the index), ``arange`` the int32 constant
    # ``1, 2, ..``, ``zero_count`` the zero-weight row count (read back with the gradient-row count) and ``zero_rows``
    # the formed index
    zero_pos: Optional[torch.Tensor] = field(default=None, repr=False)
    arange: Optional[torch.Tensor] = field(default=None, repr=False)
    zero_count: int = 0
    zero_rows: Optional[torch.Tensor] = field(default=None, repr=False)
    split: bool = False

    def finish(self) -> int:
        """Wait for the count -- the compaction kernels and the 4-byte copy (8 bytes: both counts of a zero-weight
        split); chunk 0's GEMM is already queued -- and read it (once)."""
        if self.count is None:
            self.event.synchronize()
            self.count = int(self.count_host[0])
            if self.split:
                self.zero_count = int(self.count_host[1])
        return self.count

    def zero_index(self) -> Optional[torch.Tensor]:
        """The ascending int64 index of the zero-weight rows of a zero-weight split (``None``: no such row, or no
        split), formed on the device from the counts already read back (no synchronization); the scan it reads goes
        back to the allocator once the index is formed."""
        # the one readback of both counts (the gradient-row index usually read it first)
        if self.count is None:
            self.finish()
        if self.zero_pos is not None:
            if self.zero_count:
                self.zero_rows = torch.searchsorted(
                    self.zero_pos, self.arange[: self.zero_count]
                )
            self.zero_pos = None
        return self.zero_rows

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
    path's fixed host cost behind chunk 0's logits GEMM and run 5-16 % slower with it), an ``X`` the GEMM reads in
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
    rows ``(labels >= 0) & (loss_weights != 0)`` (``valid`` holds their mask); the zero-weight row count travels in
    the same pinned copy (8 bytes) and :meth:`_HiddenCount.zero_index` forms their index after the readback."""
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
    split = loss_weights is not None
    count_host, event = (_hidden_count_pair_cell if split else _hidden_count_cell)(
        index
    )
    ar = _hidden_count_arange(index, T)
    valid = torch.empty((T,), dtype=torch.bool, device=device)
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
        # the gradient-row mask goes to ``valid`` (the fused dX finalize's mask); the zero-weight mask is a temporary
        gradient_index(
            labels,
            loss_weights,
            grad=valid,
            zero=torch.empty((T,), dtype=torch.bool, device=device),
            pos=pos,
            ar=ar,
            idx_full=idx_full,
        )
        count_host.copy_(
            pos[T - 1 :: T], non_blocking=True
        )  # both counts in one copy: gradient rows, zero-weight rows
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
        split=split,
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


# --------------------------------------------------------------------------- zero-weight split (loss entry)


def gradient_index(
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    grad: torch.Tensor,
    zero: torch.Tensor,
    pos: torch.Tensor,
    ar: torch.Tensor,
    idx_full: Optional[torch.Tensor] = None,
) -> None:
    """The zero-weight split of a policy call on the device (no host synchronization), into pre-allocated outputs:
    ``grad`` = the gradient-row mask ``G = (labels >= 0) & (loss_weights != 0)`` (a NaN weight is in ``G``), ``zero`` =
    the zero-weight mask ``Z = (labels >= 0) & (loss_weights == 0)`` (``+0`` and ``-0``), ``pos[:T]`` / ``pos[T:]`` =
    the inclusive int32 scans of ``grad`` / ``zero`` (``pos[T - 1] = |G|``, ``pos[2T - 1] = |Z|``: both counts in one
    strided copy) and, when given, ``idx_full[k] = searchsorted(pos[:T], k + 1)`` -- the ``k``-th gradient row for
    ``k < |G|``, ``T`` after it (the hidden valid-row count's gather index).  The ``k``-th zero-weight row is
    ``searchsorted(pos[T:], k + 1)`` (:meth:`_HiddenCount.zero_index`, :func:`gradient_rows`).  Two separate masks, so
    the gradient-row mask kept for the fused dX finalize holds ``T`` bytes, as the valid-row mask of the plain path."""
    T = int(labels.numel())
    valid = labels >= 0
    torch.logical_and(valid, loss_weights != 0, out=grad)
    torch.logical_xor(valid, grad, out=zero)
    torch.cumsum(grad, 0, dtype=torch.int32, out=pos[:T])
    torch.cumsum(zero, 0, dtype=torch.int32, out=pos[T:])
    if idx_full is not None:
        torch.searchsorted(pos[:T], ar[:T], out=idx_full)


# The smallest call (rows ``T``) that splits off its zero-weight rows (:func:`set_zero_weight_min_rows`, no environment
# variable): below it the call keeps every valid row on the gradient path (the split adds row kernels and a second
# count to every call that takes it).  Default 4096 (one default chunk): the floor timed for the non-deterministic
# package's split of the same loss entry (B200 and GB300, policy rows at T = 512 to 3000, no and 16.6 % zero-weight
# rows: the split did not pay off below one default chunk).  This package runs the same kernels on the gradient rows
# and adds the exact loss's term placement and reduction (:func:`set_zero_weight_loss_mode`), so the floor is re-timed
# for this package before it is relied on.
ZERO_WEIGHT_MIN_ROWS_DEFAULT = DEFAULT_CHUNK_SIZE
_ZERO_WEIGHT_MIN_ROWS = {"rows": ZERO_WEIGHT_MIN_ROWS_DEFAULT}


def zero_weight_min_rows() -> int:
    """The smallest call (rows) that splits off its zero-weight rows (:func:`set_zero_weight_min_rows`; default
    :data:`ZERO_WEIGHT_MIN_ROWS_DEFAULT`)."""
    return _ZERO_WEIGHT_MIN_ROWS["rows"]


def set_zero_weight_min_rows(rows: int) -> int:
    """Set the smallest call (rows ``T``) that splits off its zero-weight rows for the following calls of this process
    (``0``: every compacted policy call) and return the previous value.  Read once per call; no binding depends on it
    (the split changes the call's row count of the chunk loop, which the binding key holds)."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError(
            f"zero-weight min rows must be a non-negative integer, got {rows!r}"
        )
    previous = _ZERO_WEIGHT_MIN_ROWS["rows"]
    _ZERO_WEIGHT_MIN_ROWS["rows"] = int(rows)
    return previous


# How a zero-weight split forms its loss (:func:`set_zero_weight_loss_mode`, no environment variable):
# * ``"exact"`` (the default): the bits of the same call without the split.  The chunk loop keeps every chunk's per-row
#   loss terms (chunk 0's in the chunk-local ``term`` buffer, chunk ``c >= 1``'s in the compact ``loss_weights`` rows
#   of chunk ``c - 1``, which no launch reads again: no extra buffer, :func:`_term_rows`); after the loop the terms go
#   to their rows' positions among the valid rows, a zero-weight row's slot holding its weight (``+0`` / ``-0``, the
#   term ``w * min(ratio, 2)`` the plain path's ``row_finalize`` forms for any ``ratio``, NaN included), and the
#   shipped ``loss_reduce`` runs over the plain path's chunks of the valid rows (same ``rows_c``, first / last flags,
#   FP64 accumulator and FP32 rounding) -- :func:`_zero_weight_exact_loss`.
# * ``"regroup"``: the chunk loop's own ``loss_reduce`` over the gradient rows (the same FP64 sum of the same terms over
#   the split's chunk boundaries, so the loss may differ from the plain path's in its last FP32 bits; no term
#   placement, no extra launch).
ZERO_WEIGHT_LOSS_MODES = ("exact", "regroup")
_ZERO_WEIGHT_LOSS_MODE = {"mode": "exact"}


def zero_weight_loss_mode() -> str:
    """How a zero-weight split forms its loss (:func:`set_zero_weight_loss_mode`; default ``"exact"``)."""
    return _ZERO_WEIGHT_LOSS_MODE["mode"]


def set_zero_weight_loss_mode(mode: str) -> str:
    """Set how the following zero-weight splits of this process form their loss -- ``"exact"`` (the bits of the call
    without the split) or ``"regroup"`` (the split's own chunk sums, :data:`ZERO_WEIGHT_LOSS_MODES`) -- and return the
    previous mode.  Read once per call; no binding depends on it."""
    if mode not in ZERO_WEIGHT_LOSS_MODES:
        raise ValueError(
            f"zero-weight loss mode must be one of {ZERO_WEIGHT_LOSS_MODES}, got {mode!r}"
        )
    previous = _ZERO_WEIGHT_LOSS_MODE["mode"]
    _ZERO_WEIGHT_LOSS_MODE["mode"] = mode
    return previous


def zero_weight_split(
    problem: "Problem", compact: bool, min_rows: Optional[int] = None
) -> bool:
    """Whether a loss-entry call splits off its zero-weight rows: a compacted policy call of at least ``min_rows`` rows
    (``None`` = :func:`zero_weight_min_rows`)."""
    floor = zero_weight_min_rows() if min_rows is None else int(min_rows)
    return bool(
        compact
        and problem.entry == "loss"
        and problem.objective == "policy"
        and int(problem.num_rows) >= floor
    )


def _split_arange(device: torch.device, n: int) -> torch.Tensor:
    """An int32 ``1, 2, .., >= n`` on ``device``: the calling thread's cached constant on CUDA
    (:func:`_hidden_count_arange`), a fresh ``arange`` elsewhere (the reference backend on the host)."""
    if device.type == "cuda":
        return _hidden_count_arange(
            int(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ),
            n,
        )
    return torch.arange(1, n + 1, dtype=torch.int32, device=device)


def gradient_rows(
    labels: torch.Tensor, loss_weights: torch.Tensor, *, mask: bool = False
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """:func:`valid_rows` of a policy call that splits off its zero-weight rows: ``(row_index, row_valid,
    zero_index)``.  ``row_index`` / ``row_valid`` are :func:`valid_rows`' outputs for the gradient rows ``G = (labels
    >= 0) & (loss_weights != 0)`` (a NaN weight is in ``G``; ``None`` when every row is in ``G``), ``zero_index`` the
    ascending int64 index of the zero-weight rows ``Z = (labels >= 0) & (loss_weights == 0)`` (``None`` when there is
    none).  One device synchronization, as :func:`valid_rows`: a CUDA call of at most :data:`HOST_COUNT_MAX_ROWS` rows
    reads both masks back (``2T`` bytes) and forms both indices on the host; a longer one reads both counts back in one
    8-byte copy and forms both indices on the device at their exact sizes (:func:`gradient_index`; ``8 |G|`` and
    ``8 |Z|`` bytes, the ``8 T_v`` of the plain path's index)."""
    T = int(labels.numel())
    if T == 0:
        return None, None, None
    device = labels.device
    if T <= HOST_COUNT_MAX_ROWS and labels.is_cuda:
        valid = labels >= 0
        grad = valid & (loss_weights != 0)
        both = torch.stack((grad, valid & ~grad)).cpu()  # the one synchronization
        g_host, z_host = both[0].nonzero().squeeze(1), both[1].nonzero().squeeze(1)
        if int(g_host.numel()) == T:
            return None, None, None
        # pageable host memory: the copies are staged before they return, so the host indices may go right away
        return (
            g_host.to(device, non_blocking=True),
            grad if mask else None,
            z_host.to(device, non_blocking=True) if z_host.numel() else None,
        )
    grad = torch.empty((T,), dtype=torch.bool, device=device)
    pos = torch.empty((2 * T,), dtype=torch.int32, device=device)
    ar = _split_arange(device, T)
    gradient_index(
        labels,
        loss_weights,
        grad=grad,
        zero=torch.empty((T,), dtype=torch.bool, device=device),
        pos=pos,
        ar=ar,
    )
    count, zeros = (int(v) for v in pos[T - 1 :: T].tolist())  # the one synchronization
    if count == T:
        return None, None, None
    row_index = torch.searchsorted(pos[:T], ar[:count])
    zero_index = torch.searchsorted(pos[T:], ar[:zeros]) if zeros else None
    return row_index, (grad if mask else None), zero_index


def _valid_rows_of_count(
    labels: torch.Tensor, count: int, *, mask: bool = False
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """:func:`valid_rows` of a call whose valid-row count ``count`` is already on the host (a zero-weight split's two
    counts, :func:`zero_weight_plan` declined): the same index formed on the device at its exact size, no
    synchronization.  ``(None, None)`` when every row is valid."""
    T = int(labels.numel())
    if int(count) == T:
        return None, None
    valid = labels >= 0
    pos = torch.cumsum(valid, 0, dtype=torch.int32)
    index = torch.searchsorted(
        pos, _split_arange(labels.device, int(count))[: int(count)]
    )
    return index, (valid if mask else None)


def zero_weight_memory(
    *,
    num_rows: int,
    grad_rows: int,
    zero_rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    stats_tile: int = GEOMETRY_DEFAULTS["stats_tile"],
    need_dx: bool = True,
    need_dw: bool = True,
    fuse_dw_cast: bool = False,
    dx_finalize: bool = False,
    hidden_count: bool = False,
    return_logp: bool = True,
    exact_loss: bool = False,
    plain_dx_ws_slabs: int = 0,
    split_dx_ws_slabs: int = 0,
    pass_arena: bool = False,
) -> dict[str, Any]:
    """The byte model of a loss call with ``zero_rows`` zero-weight rows and ``grad_rows`` gradient rows, with the split
    and without it (host arithmetic on the counts, deterministic): the memory rule's guard (:func:`zero_weight_plan`).
    Both sides count the bytes above the caller's tensors that differ between them (the outputs are the same).

    * ``plain_bytes`` = the chunk loop over the ``T_v = |G| + |Z|`` valid rows: :func:`memory_report`'s temporaries
      plus accumulators (``plain_dx_ws_slabs``: that call's K-slice slabs).
    * the split's phases: ``loop`` = the same report over ``|G|`` rows (``split_dx_ws_slabs``) plus the zero-weight
      index ``8 |Z|`` (plus the pass's compact ``logp`` ``4 |Z|`` when it ran first); ``pass`` = the logp-only pass's
      buffers ``P = workspace_layout(|Z|, logprob, compact) + 8 |Z|`` (its ``lse`` and compact ``logp``) next to the
      held indices ``H_i = 8 |G| + 8 |Z| (+ T row_valid)`` -- first: plus chunk 0's ``logits`` / ``stats`` / ``x_c``
      when the hidden valid-row count holds them; after the loop: plus the accumulators, the deferred chunk's ``dz``
      rows and the loss cell (``A``); ``exact`` (the exact loss) = ``H_i + A`` plus ``4 T_v`` (``term_v``),
      ``max(8 |G|, 12 |Z|)`` (positions, weights) and the two kept term buffers ``4 |G| + 4 min(C, |G|)``.
    * ``zero_first`` -- where the pass runs: without the hidden count ahead of the loop, except after it when the pass
      would take its own chunk buffers (no ``pass_arena``) and the loop leaves no chunk alive until the backward (it
      reuses the loop's freed chunk buffers, e.g. the gradient-free first forward of a checkpointed step); with the
      hidden count after the loop when no chunk loop follows or no chunk stays alive until the backward; with both (the
      deferred last chunk) the order of the lower split peak, ahead of the loop on a tie (``orders``: both peaks; empty
      when the order is fixed).
    * ``pass_arena`` (the templated binding: the caller's ``backend == "cake"``): a pass ahead of a loop that keeps
      ``dW_acc`` takes its chunk buffers (``logits`` / ``stats`` / ``x_c``) from that FP32 accumulator, allocated ahead
      of the pass, when they fit in it (:func:`_arena_views`; the result's ``pass_arena``): that phase then holds
      ``dW_acc`` and the pass's other buffers instead of the carved ones.  A pass of its own chunk buffers there would
      leave a freed segment of ``min(C, |Z|)`` rows that the loop's larger allocations (its chunk buffers, ``dW_acc``,
      the backward's ``dW``) cannot reuse, a reserved peak above the plain call's.
    * ``split_bytes`` = the largest phase in that order; ``split`` = ``split_bytes <= plain_bytes``."""
    T, n_g, n_z = int(num_rows), int(grad_rows), int(zero_rows)
    n_v = n_g + n_z
    if not (n_g >= 0 and n_z > 0 and n_v <= T):
        raise ValueError(
            f"zero_weight_memory: 0 <= grad_rows, 0 < zero_rows and grad_rows + zero_rows <= num_rows, got "
            f"{n_g}, {n_z}, {T}"
        )
    fuse = bool(need_dw and fuse_dw_cast)
    fused_dx = bool(need_dx and dx_finalize)

    def loop(rows: int, slabs: int) -> dict[str, Any]:
        return memory_report(
            T,
            hidden,
            vocab,
            chunk,
            stats_tile=stats_tile,
            entry="loss",
            need_dx=need_dx,
            need_dw=need_dw,
            return_logp=True,
            dx_ws_slabs=int(slabs) if need_dx else 0,
            valid_rows=None if rows == T else rows,
            fuse_dw_cast=fuse,
            dx_finalize=fused_dx,
            hidden_count=hidden_count,
        )

    plain_m = loop(n_v, plain_dx_ws_slabs)
    plain = int(plain_m["temporary_bytes"]) + int(plain_m["accumulator_bytes"])
    if n_g:
        split_m = loop(n_g, split_dx_ws_slabs)
        loop_bytes = int(split_m["temporary_bytes"]) + int(split_m["accumulator_bytes"])
        acc = int(split_m["accumulator_bytes"])
        saved_dz = int(split_m["saved_dz_bytes"])
        chunk0 = sum(int(split_m["temporary"].get(k, 0)) for k in _ZERO_PASS_CHUNK0)
    else:
        loop_bytes = acc = saved_dz = chunk0 = 0
    held = 8 * n_g + 8 * n_z + (T if fused_dx else 0)
    pass_layout = workspace_layout(
        n_z,
        vocab,
        chunk,
        stats_tile=stats_tile,
        entry="logprob",
        hidden=hidden,
        compact=True,
    )
    pass_bytes = pass_layout["total"] + 8 * n_z
    carved = _carved_bytes(pass_layout)
    dw_acc = int(split_m["accumulators"].get("dW_acc", 0)) if n_g else 0
    arena = bool(pass_arena and dw_acc and carved <= dw_acc)
    after = held + acc + saved_dz + 4
    exact_bytes = (
        4 * n_v + max(8 * n_g, 12 * n_z) + (4 * n_g + 4 * min(int(chunk), n_g))
        if exact_loss
        else 0
    )

    def peak(first: bool) -> tuple[int, dict[str, int]]:
        z_hold = 4 * n_z if (first and return_logp) else 0
        phases: dict[str, int] = {}
        if n_g:
            phases["loop"] = loop_bytes + 8 * n_z + z_hold
        if return_logp:
            phases["pass"] = (
                pass_bytes
                + held
                + (chunk0 if hidden_count else 0)
                + (dw_acc - carved if arena else 0)
                if first
                else pass_bytes + after
            )
        if exact_loss:
            phases["exact"] = after + z_hold + exact_bytes
        return max(phases.values(), default=0), phases

    orders: dict[str, int] = {}
    if not hidden_count:
        # ahead of the loop, unless it would take its own chunk buffers (no arena) and the loop leaves none alive
        # (no deferred last chunk): then after it, in the loop's freed chunk buffers
        first = bool(n_g == 0 or arena or saved_dz)
    elif n_g == 0 or not fuse:
        first = False
    else:
        orders = dict(first=peak(True)[0], after=peak(False)[0])
        first = not orders["after"] < orders["first"]
    split_bytes, phases = peak(first)
    return dict(
        plain_bytes=plain,
        split_bytes=split_bytes,
        phases=phases,
        orders=orders,
        zero_first=bool(first and return_logp),
        pass_arena=bool(arena and first and return_logp),
        split=split_bytes <= plain,
    )


def _dx_slice_rule(
    problem: "Problem",
    device: torch.device,
    *,
    backend: str,
    need_dx: bool,
    tuning: dict,
):
    """``(stats_tile, rule)`` of a loss call: ``rule(rows, cap=None)`` the largest dX slice count of a chunk loop of
    ``rows`` rows on ``device``'s registered program (:func:`dx_slices_max`, the arguments :func:`prepare_lm_head_det_loss`
    plans with); 1 on the reference backend or without ``dX``."""
    record = module_name = None
    if backend == "cake":
        module_name, record = record_for(device)
    geometry = Geometry.from_record(record)
    if record is None or not need_dx:
        return int(geometry.stats_tile), lambda rows, cap=None: 1
    stages = registered_stages(module_name)
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    dx_tuning = tuning.get("dx", {})
    explicit_k = (
        (1 if dx_tuning["k_slices"] is None else int(dx_tuning["k_slices"]))
        if "k_slices" in dx_tuning
        else None
    )
    rule = functools.partial(
        dx_slices_max,
        hidden=int(problem.hidden),
        vocab=int(problem.vocab),
        chunk=int(problem.chunk),
        num_sms=num_sms,
        dx_max=dx_max_slices(stages),
        geometry=geometry,
        resident=cluster_resident(device, geometry.dx_cluster_ctas, num_sms),
        explicit_k=explicit_k,
    )
    return int(geometry.stats_tile), lambda rows, cap=None: rule(int(rows), cap=cap)


def zero_weight_plan(
    problem: "Problem",
    device: torch.device,
    *,
    grad_rows: int,
    zero_rows: int,
    backend: str,
    need_dx: bool,
    need_dw: bool,
    fuse_dw_cast: bool,
    dx_finalize: bool,
    hidden_count: bool,
    return_logp: bool,
    exact_loss: bool,
    tuning: dict,
) -> dict[str, Any]:
    """The memory rule of a zero-weight split, decided on the host from the call's two counts (no further
    synchronization): the split's chunk loop never runs more dX K slices per chunk than the call without the split
    (``dx_slices_cap``: that call's largest count, or ``None`` when the split's own rule stays within it), and the
    call splits only when :func:`zero_weight_memory`'s predicted split peak is at most the call's predicted peak
    without the split (``split``; ``zero_first``: where the logp-only pass runs)."""
    n_g, n_z = int(grad_rows), int(zero_rows)
    stats_tile, rule = _dx_slice_rule(
        problem, device, backend=backend, need_dx=need_dx, tuning=tuning
    )
    plain_slices = rule(n_g + n_z)
    split_slices = rule(n_g) if n_g else 1
    cap = None
    if split_slices > plain_slices:
        cap = plain_slices
        split_slices = rule(n_g, cap=cap)
    model = zero_weight_memory(
        num_rows=int(problem.num_rows),
        grad_rows=n_g,
        zero_rows=n_z,
        hidden=int(problem.hidden),
        vocab=int(problem.vocab),
        chunk=int(problem.chunk),
        stats_tile=stats_tile,
        need_dx=need_dx,
        need_dw=need_dw,
        fuse_dw_cast=fuse_dw_cast,
        dx_finalize=dx_finalize,
        hidden_count=hidden_count,
        return_logp=return_logp,
        exact_loss=exact_loss,
        plain_dx_ws_slabs=plain_slices - 1,
        split_dx_ws_slabs=split_slices - 1,
        pass_arena=backend == "cake",
    )
    model["dx_slices_cap"] = cap
    model["plain_dx_slices"] = plain_slices
    model["split_dx_slices"] = split_slices
    return model


# --------------------------------------------------------------------------- zero-gradient row skip


# The smallest log-probability backward (rows of its chunk loop) that runs the zero-gradient row skip
# (:func:`set_zero_dlogp_min_rows`, no environment variable; ``None`` = never).  The skip orders the rows on the device
# (one partition launch per call, no host synchronization) at the start of the backward, where the GPU waits for the
# host.  The generic floor is one row above one default chunk: a smaller call keeps the backward without the skip on an
# architecture without a floor of its own (:data:`ZERO_DLOGP_MIN_ROWS_BY_ARCH`).
ZERO_DLOGP_MIN_ROWS_DEFAULT: Optional[int] = DEFAULT_CHUNK_SIZE + 1
# Per-architecture default floors (valid rows); an architecture not listed (sm_107a included) uses
# :data:`ZERO_DLOGP_MIN_ROWS_DEFAULT`.  The values are those timed for the non-deterministic package's skip (plain and
# temperature log-probability backward, step and checkpointed step, with no zero-gradient row: no slower than the
# same head without the skip at the floor): B200 (sm_100a) 1946 valid rows (T = 2048 with 5 % ignored rows), GB300
# (sm_103a) 3891 valid rows (T = 4096 with 5 % ignored rows).  This package runs the same counted dz / dW forms, but
# its dX GEMM keeps its uncounted forms (it computes every row of a chunk), so the floors are re-timed for this
# package before they are relied on.
ZERO_DLOGP_MIN_ROWS_BY_ARCH: dict[str, int] = {"sm_100a": 1946, "sm_103a": 3891}


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
    """The partition kernel's four outputs: int64 ``order [rows]``, int32 ``scan [T]`` and ``counts [num_chunks]`` split
    from ONE int32 allocation (each region at a :data:`WORKSPACE_ALIGN`-byte boundary, :func:`_align`), and FP32
    ``lse_out [rows]`` on its own.  ``order`` and ``scan`` live until the dX output boundary (:func:`finalize_dx`);
    ``lse_out`` is read by the chunk loop only, so a separate allocation lets the backward release it before the
    output boundary allocates ``dX`` (the boundary holds 4 bytes per valid row less)."""
    o_scan = _align(8 * rows) // 4
    o_counts = o_scan + _align(4 * T) // 4
    block = torch.empty((o_counts + num_chunks,), dtype=torch.int32, device=device)
    order, _, scan, _, counts = block.split_with_sizes(
        (2 * rows, o_scan - 2 * rows, T, o_counts - o_scan - T, num_chunks)
    )
    lse_out = torch.empty((rows,), dtype=torch.float32, device=device)
    return order.view(torch.int64), lse_out, scan, counts


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
    ``backend="cake"`` this is ONE launch: the reference's small kernels each pay a host launch at the start of the
    backward, ahead of its first GEMM.  The kernel reads ``labels`` for the valid rows instead of ``row_index``, which
    must be the ascending index of the rows with a non-negative label (the forward's valid-row index, as the backward
    passes it).  Any other call (another device or backend, a record without the kernel, an int64 / FP32 contract the
    inputs do not meet) runs the reference.  The ordering is a stable partition formed from an inclusive scan, so it is
    a pure function of ``labels`` and of which ``dlogp`` entries are zero: the same outputs on every run.

    The call sits on the host path ahead of the backward's first launch, where the GPU waits: the device's record is
    resolved once (:func:`_skip_record`), ``order`` / ``scan`` / ``counts`` are one allocation
    (:func:`_partition_outputs`) and, while :data:`BINDING_CACHE` is enabled, the templated launch of each ``(device,
    T, chunk)`` form is remembered with the eager output-boundary launches (:func:`_eager_launch`'s cache, under a key
    of those values) -- a later call only slots its tensors into it."""
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
        key = (ZERO_DLOGP_PARTITION_STAGE, "eager")
        launch = _bind_all(
            record,
            module_name,
            (key,),
            {key: values},
            device,
            Geometry.from_record(record),
        )[key]
        arguments = list(launch.arguments)
        if cache.enabled:
            cache.eager_remember(form, record, launch.templated())
    else:
        arguments = launch.arguments_for(tensors)
    with _ffi_stream_context(int(index)):
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
    live = torch.logical_and(labels.reshape(-1) >= 0, dlogp.reshape(-1) != 0)
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
    with the ``dz`` arm and, when it produces ``dX``, the fused dX finalize (the only reader of ``dX_acc``: it reads the
    live rows through the partition's scan and writes the skipped rows' exact zeros, so the rows past a chunk's count,
    whose ``dX_acc`` rows the uncounted dX GEMM forms from stale ``dz``, are never read), on a record that registers the
    counted dz arm (:data:`LOGPROB_DZ_DCNT_STAGE`) and the partition kernel (:data:`ZERO_DLOGP_PARTITION_STAGE`).  An
    eligible call runs the skip when its plan has the counted forms (:attr:`Plan.dcnt`), and the backward without it
    otherwise.  The floor is the device's (:func:`zero_dlogp_min_rows` of its index: the override, else its
    architecture's default)."""
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
    "dx_reduce": _dx_reduce,
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
    a compacted log-probability backward scatters it in one pass, :func:`finalize_dx`).  The ``slab_sum`` kernel of a
    K-sliced dX GEMM is a per-chunk instance like the dX GEMM's ``_s`` form (:attr:`Plan.stages`), not listed here.
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
    architecture (:func:`reachable_variants`)."""
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


def validate_lm_head_det_inputs(
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


def plan_chunks(num_rows: int, chunk: int) -> tuple[tuple[int, int], ...]:
    """``(row0, rows_c)`` of every chunk in launch order: full chunks first, the
    tail (``T % C``) last, never dropped, never padded; empty for ``T == 0``."""
    if int(chunk) < 1:
        raise ValueError("chunk must be positive")
    return tuple(
        (row0, min(int(chunk), int(num_rows) - row0))
        for row0 in range(0, int(num_rows), int(chunk))
    )


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
    ] = ()  # K-slice count of the dX GEMM per chunk (empty = one slice everywhere)
    valid_rows: Optional[int] = (
        None  # rows of the compacted chunk loop (the valid rows); None = every row of the caller
    )
    fuse_dw_cast: bool = False  # the last chunk's weight-gradient GEMM runs in the backward with the fused scale + cast
    variants: tuple[
        ChunkVariants, ...
    ] = ()  # per chunk: the GEMM instance knobs the rules resolved (empty = defaults)
    dx_finalize: bool = False  # a K-sliced dX GEMM's slabs are added by the ``slab_sum`` kernel (else the host's ``add_`` chain)
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
    # the zero-gradient row skip of the log-probability backward (:data:`DCNT_BASES`): the dz arm and the weight-gradient
    # GEMMs run their ``_dcnt`` forms, bounded by the per-chunk device counts ``t["dcnt"]`` of the rows the call ordered
    # first (:func:`zero_dlogp_partition`); the dX GEMM keeps its forms (every row of the chunk)
    dcnt: bool = False
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
        """The log-probability backward's recompute GEMM of chunk ``index``: the ``dz`` arm (:attr:`logprob_dz`; its
        ``_dcnt`` form under the zero-gradient row skip, :attr:`dcnt`), else the stats-free logits GEMM that
        ``row_grad`` follows."""
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
        """The dX GEMM form of chunk ``index`` (K-sliced or not, its tile and ring depth; the slice count is a launch
        scalar, :meth:`dx_slices_of`)."""
        v = self.variants_of(index)
        return dx_stage(
            self.dx_slices_of(index),
            v.dx_tile_n,
            v.dx_stages,
            split=v.dx_split is not None,
            swab=v.dx_swab,
        )

    def dw_acc_stage(self, index: int) -> str:
        """The weight-gradient accumulate form of chunk ``index`` (blocked when the block rule set a ``group_n``; the
        raster height and the block width are launch scalars; counted under the zero-gradient row skip)."""
        v = self.variants_of(index)
        return stage_variant(
            "gemm_dw_acc",
            blocked=v.dw_group_n is not None,
            epi_store=v.dw_epi_store,
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
    def dx_ws_slabs(self) -> int:
        """FP32 ``[rows, H]`` slabs of the K-slice workspace the plan needs (largest slice count - 1)."""
        return int(max(self.dx_slices, default=1)) - 1 if self.need_dx else 0

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
        if self.need_dx and self.dx_finalize and self.dx_ws_slabs:
            used.add(
                "slab_sum"
            )  # some chunk slices its dX GEMM: its slabs are added by the kernel
        for index in range(self.num_chunks):
            if "gemm_logits" in base:
                used.add(self.forward_logits_stage(index))
            if "gemm_logits_nostats" in base:
                used.add(self.recompute_stage(index))
            if "gemm_dx" in base:
                used.add(self.dx_stage_of(index))
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
                dcnt=self.dcnt,  # the counted form under the zero-gradient row skip (bf16 only)
            )
        return "scale_cast_f32" if fp32 else "scale_cast_bf16"


class _TemperaturePlan(Plan):
    """A :class:`Plan` with the temperature forms (:data:`TEMPERATURE_STAGES`, :func:`make_plan_temperature`: a
    log-probability plan of a registered program whose call has a temperature, which then runs both logits GEMM
    arms): the forward's logits GEMM, chunk 0's device-count GEMM, ``row_finalize`` and the backward's ``dz``
    recompute run their ``_tau`` forms.  The reference path keeps the plan without them and reads the temperature
    from the host values (:attr:`Problem.temperature`).  Under the zero-gradient row skip (:attr:`Plan.dcnt`,
    :func:`make_plan_temperature`) this plan is the mixed one: the ``dz`` recompute runs its uncounted temperature form
    over every row of the chunk (a zero-gradient row's ``dz`` is zero: ``RN(0 / tau) = 0``), the weight-gradient GEMMs
    their ``_dcnt`` forms; :class:`_CountedTemperaturePlan` counts the ``dz`` recompute too."""

    temperature: ClassVar[bool] = True
    # the plan class without the temperature forms (:meth:`without_temperature`)
    plain: ClassVar[type] = Plan
    # whether the ``dz`` recompute runs its counted temperature form under the skip (:class:`_CountedTemperaturePlan`)
    counted_dz: ClassVar[bool] = False

    def without_temperature(self) -> Plan:
        """The same plan without the temperature forms: the twin on which :func:`stage_values_temperature` checks a
        temperature form.  The twin of the mixed plan's uncounted ``dz`` temperature form is the uncounted
        ``gemm_logits_dz``, so the twin keeps :attr:`Plan.dcnt` off; the counted dW forms of a plan under the
        zero-gradient row skip are checked on the plan itself (:class:`_CountedTemperaturePlan` keeps it on)."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["dcnt"] = False
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
        if self.need_dx and self.dx_finalize and self.dx_ws_slabs:
            used.add(
                "slab_sum"
            )  # some chunk slices its dX GEMM: its slabs are added by the kernel
        for index in range(self.num_chunks):
            if "gemm_logits" in base:
                used.add(self.forward_logits_stage(index))
            if "gemm_logits_nostats" in base:
                used.add(self.recompute_stage(index))
            if "gemm_dx" in base:
                used.add(self.dx_stage_of(index))
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
    """The kept-logits plan of the log-probability entry (:func:`cake_lm_head_det_train_logprob_keep`): a :class:`Plan`
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
    logprob_keep: bool = False,
    dcnt: bool = False,
    counted_tma_cast: bool = True,
    dx_slices_cap: Optional[int] = None,
) -> Plan:
    """The chunk schedule; ``dx_max_slices`` > 1 (K-sliced dX stages registered) picks each chunk's slice count
    (``dx_resident`` = the device's co-resident dX clusters, :func:`cluster_resident`); ``valid_rows`` (compaction)
    chunks that many rows instead of ``T``; ``fuse_dw_cast`` defers the last chunk's weight-gradient GEMM to the
    backward with the fused scale + cast epilogue.  ``arch`` (a registered program's architecture) applies the
    per-chunk instance rules (:func:`chunk_variants`; ``None`` = the reference path, base stages only) and
    ``gemm_tuning`` pins knobs explicitly (already validated by the caller or :func:`_resolve_gemm_tuning`);
    ``dx_finalize`` adds a sliced dX GEMM's slabs with the ``slab_sum`` kernel and ``dx_cast=False`` leaves ``dx_acc``
    uncast for the caller (:func:`finalize_dx`).  ``hidden_count`` (the eager entry points' hidden valid-row count,
    ``arch`` required): chunk 0's gather and logits GEMM already ran in device-count form (``hidden_stats``: with the
    statistics) at the buffer extent ``min(chunk, T)``, whose raster-rule height the plan records.  ``logprob_dz`` /
    ``logprob_lbl`` (a log-probability plan of a record that registers them) run the entry's logits GEMM arms
    (:attr:`Plan.logprob_dz`, :attr:`Plan.logprob_lbl`).  ``logprob_keep`` (a log-probability plan, a registered
    program): the kept-logits plan (:class:`_KeptPlan`), which runs neither arm.  ``dcnt`` (a log-probability plan
    with the dz arm, a registered program, not the hidden valid-row count): the zero-gradient row skip's ``_dcnt``
    forms (:attr:`Plan.dcnt`) for the dz arm and the weight-gradient GEMMs -- the last chunk's bf16 fused cast in the
    TMA epilogue's counted form where :func:`dw_cast_tma_rule` takes the TMA epilogue (``counted_tma_cast=False``, a
    record without that form: the shipped cast's counted form instead; the same ``dW`` bits).  The chunks, the dX
    slice counts and the dX forms are the plan's without the skip.  Off when a chunk takes a form without a counted
    one (an explicitly pinned weight-gradient TMA epilogue, an fp32 fused cast).  ``dx_slices_cap`` (the zero-weight
    split of the loss entry, :func:`forward_loss`): no chunk's rule-picked dX slice count exceeds it
    (:func:`capped_k_slices`; an explicit ``k_slices`` knob is not capped)."""
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
    chunks = plan_chunks(rows, problem.chunk)
    tuning = gemm_tuning or {}
    dx_tuning = tuning.get("dx", {})
    slices: tuple[int, ...] = ()
    if need_dx and "k_slices" in dx_tuning:
        # an explicit knob wins over the slice rule: a value pins that slice count, ``None`` pins the default form
        # (one slice, the base ``gemm_dx`` stage) -- the same ``None`` semantics as every other pinned knob
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
        slices = tuple(explicit_k for _ in chunks)
    elif need_dx and int(dx_max_slices) > 1:
        slices = tuple(
            capped_k_slices(
                recommended_k_slices(
                    rows_c,
                    problem.hidden,
                    num_sms,
                    dx_max_slices,
                    geometry,
                    dx_resident,
                    vocab=problem.vocab,
                ),
                dx_slices_cap,
                problem.vocab,
                geometry,
            )
            for _, rows_c in chunks
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
    variants: tuple[ChunkVariants, ...] = ()
    if arch is not None or tuning:
        variants = tuple(
            chunk_variants(
                rows_c,
                problem.hidden,
                num_sms,
                slices[i] if slices else 1,
                geometry,
                arch,
                tuning,
                num_chunks=len(chunks),
                ce_loss=problem.entry == "loss" and problem.objective == "ce",
            )
            for i, (_, rows_c) in enumerate(chunks)
        )
    fp32_cast = problem.entry == "loss" and problem.grad_weight_dtype == torch.float32
    # the zero-gradient row skip's counted forms: a log-probability plan with the dz arm on a registered program; none
    # for an explicitly pinned weight-gradient TMA epilogue (the accumulate's TMA reduce-add has no counted form) or an
    # fp32 fused cast.  The dX GEMM keeps its forms: no dX knob decides the skip.
    dcnt = bool(
        dcnt
        and arch is not None
        and problem.entry == "logprob"
        and logprob_dz
        and not logprob_keep
        and not hidden_count
        and not (need_dw and fp32_cast)
        and not any(v.dw_epi_store not in (None, "redsm") for v in variants)
    )
    if (
        variants
        and (not dcnt or counted_tma_cast)
        and need_dw
        and fuse_dw_cast
        and not fp32_cast
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
    )


def make_plan_temperature(
    problem: Problem, *, counted_dz: bool = True, **kw: Any
) -> Plan:
    """:func:`make_plan` of a log-probability call with a temperature (:attr:`Problem.temperature`) on a registered
    program (``arch``) running both logits GEMM arms (``logprob_dz`` and ``logprob_lbl``): the plan runs the
    temperature forms (:class:`_TemperaturePlan`).  ``dcnt`` (the zero-gradient row skip, :func:`make_plan`'s rule
    and fallbacks): the weight-gradient GEMMs run their ``_dcnt`` forms over the rows the call ordered first, and the
    ``dz`` recompute its counted temperature form (``counted_dz``, :class:`_CountedTemperaturePlan`: no work past the
    chunk's count) or, with ``counted_dz=False``, the uncounted one over every row (the mixed plan: a zero-gradient
    row's ``dz`` is ``RN(0 / tau) * (...) = 0``)."""
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
    dx_ws_slabs: int = 0,
    hidden: int = 0,
    compact: bool = False,
) -> dict:
    """Byte ``(offset, size)`` of every workspace region plus ``"total"``.

    ``dx_ws_slabs`` > 0 adds the FP32 ``dx_ws [slabs, min(T, C), H]`` slabs of
    the K-sliced dX GEMM (``hidden`` = ``H``); ``compact`` adds the BF16
    ``x_c [min(T, C), H]`` buffer the valid rows of ``X`` are gathered into
    (``num_rows`` is then the valid-row count).

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
    if dx_ws_slabs:
        if int(hidden) < 1:
            raise ValueError(
                "workspace_layout needs hidden > 0 for the K-slice workspace"
            )
        sizes.append(("dx_ws", int(dx_ws_slabs) * rows * int(hidden) * 4))
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
    dx_ws_slabs: int = 0,
    valid_rows: Optional[int] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    hidden_count: bool = False,
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
    vocabulary-sized buffer (``min(T, C)``).

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
    """
    compact = valid_rows is not None
    rows = (
        int(valid_rows) if compact else int(num_rows)
    )  # rows the chunk loop processes
    chunks = plan_chunks(rows, chunk)
    deferred = bool(need_dw and _resolve_fuse(fuse_dw_cast) and chunks)
    fused_dx = bool(need_dx and _resolve_dx_finalize(dx_finalize))
    layout = workspace_layout(
        rows,
        vocab,
        chunk,
        stats_tile=stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_workspace_bytes,
        scratch_bytes=scratch_bytes,
        dx_ws_slabs=dx_ws_slabs if need_dx else 0,
        hidden=hidden,
        compact=compact,
    )
    temporary = {
        name: size
        for name, (_, size) in ((k, v) for k, v in layout.items() if k != "total")
    }
    temporary["lse"] = rows * 4
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
                temporary=temporary,
                outputs_bytes=sum(outputs.values()),
                outputs=outputs,
                accumulator_bytes=sum(accumulators.values()),
                accumulators=accumulators,
                weights_bytes=sum(weights.values()),
                weights=weights,
                vocab_rows_max=min(int(chunk), max(rows, 1)),
                chunk=int(chunk),
                num_chunks=len(chunks),
                compact_rows=compact,
                valid_rows=rows,
                gather_bytes=int(layout["x_c"][1]) if compact else 0,
                fuse_dw_cast=deferred,
                saved_dz_bytes=min(int(chunk), rows) * int(vocab) * 2
                if deferred
                else 0,
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
        temporary=temporary,
        outputs_bytes=sum(outputs.values()),
        outputs=outputs,
        accumulator_bytes=sum(accumulators.values()),
        accumulators=accumulators,
        weights_bytes=sum(weights.values()),
        weights=weights,
        vocab_rows_max=min(int(chunk), max(rows, 1)),
        chunk=int(chunk),
        num_chunks=len(chunks),
        compact_rows=compact,
        valid_rows=rows,
        gather_bytes=int(layout["x_c"][1]) if compact else 0,
        fuse_dw_cast=deferred,
        saved_dz_bytes=min(int(chunk), rows) * int(vocab) * 2 if deferred else 0,
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
    dx_ws_slabs: int = 0,
    valid_rows: Optional[int] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    hidden_count: bool = False,
    temperature: bool = False,
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
    vocabulary-sized buffer (``min(T, C)``).

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
    chunks = plan_chunks(rows, chunk)
    deferred = bool(need_dw and _resolve_fuse(fuse_dw_cast) and chunks)
    fused_dx = bool(need_dx and _resolve_dx_finalize(dx_finalize))
    layout = workspace_layout(
        rows,
        vocab,
        chunk,
        stats_tile=stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_workspace_bytes,
        scratch_bytes=scratch_bytes,
        dx_ws_slabs=dx_ws_slabs if need_dx else 0,
        hidden=hidden,
        compact=compact,
    )
    temporary = {
        name: size
        for name, (_, size) in ((k, v) for k, v in layout.items() if k != "total")
    }
    temporary["lse"] = rows * 4
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
                temporary=temporary,
                outputs_bytes=sum(outputs.values()),
                outputs=outputs,
                accumulator_bytes=sum(accumulators.values()),
                accumulators=accumulators,
                weights_bytes=sum(weights.values()),
                weights=weights,
                vocab_rows_max=min(int(chunk), max(rows, 1)),
                chunk=int(chunk),
                num_chunks=len(chunks),
                compact_rows=compact,
                valid_rows=rows,
                gather_bytes=int(layout["x_c"][1]) if compact else 0,
                fuse_dw_cast=deferred,
                saved_dz_bytes=min(int(chunk), rows) * int(vocab) * 2
                if deferred
                else 0,
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
        temporary=temporary,
        outputs_bytes=sum(outputs.values()),
        outputs=outputs,
        accumulator_bytes=sum(accumulators.values()),
        accumulators=accumulators,
        weights_bytes=sum(weights.values()),
        weights=weights,
        vocab_rows_max=min(int(chunk), max(rows, 1)),
        chunk=int(chunk),
        num_chunks=len(chunks),
        compact_rows=compact,
        valid_rows=rows,
        gather_bytes=int(layout["x_c"][1]) if compact else 0,
        fuse_dw_cast=deferred,
        saved_dz_bytes=min(int(chunk), rows) * int(vocab) * 2 if deferred else 0,
        dx_finalize=fused_dx,
        hidden_count=False,
    )


def kept_memory_report(memory: dict[str, Any], plan: Plan) -> dict[str, Any]:
    """:func:`memory_report` of a kept-logits plan (:class:`_KeptPlan`): the forward keeps every chunk's BF16 logits for
    the backward -- chunk 0 in the chunk buffer (the ``logits`` temporary, alive until the backward), the chunks after
    it in one ``[T_v - rows_0, V]`` buffer (``accumulators["saved_logits"]``); ``saved_logits_bytes`` counts both,
    ``2 * T_v * V``.  The memory rule's chunk bound does not hold for them.  The default report has neither key."""
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


def lm_head_det_loss_workspace_size(
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
    """Workspace bytes :func:`prepare_lm_head_det_loss` needs for ``(T, V, C)`` on ``device`` (``hidden`` = ``H``,
    required for the compacted path and for a program with K-sliced dX stages).  ``compact_rows`` adds the gather buffer of the
    compacted path; sized for ``T`` rows it bounds every valid-row count -- including the K-slice slabs of a
    compacted plan, whose tail chunk (the valid-row count modulo ``C``) may take more slices than any full-``T``
    chunk (the slab count is taken over every possible tail row count as well)."""
    if compact_rows and not hidden and backend == "reference":
        raise ValueError(
            "lm_head_det_loss_workspace_size needs hidden= for the compacted path"
        )
    if backend == "reference":
        return int(
            workspace_layout(
                num_rows,
                vocab,
                chunk,
                entry=entry,
                hidden=hidden or 0,
                compact=compact_rows,
            )["total"]
        )
    name, record = record_for(device)
    stages = registered_stages(name)
    geometry = Geometry.from_record(record)
    hidden = None if hidden is None else int(hidden)
    slabs = 0
    if need_dx and dx_max_slices(stages) > 1:
        if hidden is None:
            raise ValueError(
                "lm_head_det_loss_workspace_size needs hidden= for a program with K-sliced dX stages"
            )
        num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        resident = cluster_resident(
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device),
            geometry.dx_cluster_ctas,
            num_sms,
        )
        rows_candidates = [rows_c for _, rows_c in plan_chunks(num_rows, chunk)]
        if compact_rows:
            # the valid-row count is unknown here: the compacted plan's tail chunk can have any row count up to one
            # chunk; the slice rule depends on rows only through the row-tile count, so one row count per tile suffices
            cap = min(int(num_rows), int(chunk))
            rows_candidates += list(range(1, cap + 1, int(geometry.row_tile))) + [cap]
        slabs = (
            max(
                (
                    recommended_k_slices(
                        rows_c,
                        hidden,
                        num_sms,
                        dx_max_slices(stages),
                        geometry,
                        resident,
                        vocab=vocab,
                    )
                    for rows_c in rows_candidates
                ),
                default=1,
            )
            - 1
        )
    return int(
        workspace_layout(
            num_rows,
            vocab,
            chunk,
            stats_tile=geometry.stats_tile,
            entry=entry,
            tma_workspace_bytes=_record_tma_bytes(record, stages),
            scratch_bytes=_record_scratch_bytes(record, stages),
            dx_ws_slabs=slabs,
            hidden=hidden or 0,
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


# the workspace regions (:func:`workspace_layout`) a template runner takes from a call's hidden valid-row count
# (:class:`_HiddenCount`) instead of allocating them: region -> the count's chunk-0 buffer
CHUNK_VIEW_REGIONS = {"logits": "logits", "stats": "stats_buf", "x_c": "x_c"}


def _chunk_view_layout(layout: dict) -> dict:
    """``layout`` (:func:`workspace_layout`) without the regions of :data:`CHUNK_VIEW_REGIONS`: every other region in
    the same order and alignment from offset 0 of one smaller allocation, ``"total"`` its size."""
    out: dict = {}
    offset = 0
    for name, region in layout.items():
        if name == "total" or name in CHUNK_VIEW_REGIONS:
            continue
        out[name] = (offset, region[1])
        offset += _align(region[1])
    out["total"] = offset
    return out


def _chunk_view(hidden: "_HiddenCount", name: str, shape) -> torch.Tensor:
    """Region ``name`` of :data:`CHUNK_VIEW_REGIONS` as the leading ``shape[0]`` rows of the hidden valid-row count's
    chunk-0 buffer (``min(chunk, T)`` rows, at least a runner's ``min(chunk, rows)`` for ``rows <= T``)."""
    view = getattr(hidden, CHUNK_VIEW_REGIONS[name])[: shape[0]]
    if tuple(view.shape) != tuple(shape):
        raise ValueError(
            f"the hidden valid-row count's {name} buffer {tuple(view.shape)} cannot hold the region {tuple(shape)}"
        )
    return view


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
    module = load_cake_lm_head_det_loss_module(module_name, stage)
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
    capturing stream.  The cells belong to no call, so they are made outside any active torch
    dispatch mode: selective activation checkpointing replays a recomputation's operators against
    its checkpointed forward's, and a first use inside a recomputation would add these."""
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
    chunk's ``row_grad`` key or ``dz`` recompute (:data:`LOGPROB_DZ_STAGE` or its counted form; the fork event is recorded after it) and the per-chunk ``dW`` accumulate keys (launched
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
    elif base == "gemm_dx":
        # every token chunk writes its own rows of dX_acc: the first K chunk of the GEMM always stores (first_chunk=1);
        # the K-sliced forms write slices >= 1 into the ``dx_ws`` slabs (added by ``dx_reduce`` afterwards)
        k = plan.dx_slices_of(index)
        want = plan.dx_stage_of(index)
        if stage != want:
            raise ValueError(
                f"chunk {index} plans {k} dX slice(s) as {want!r}; stage {stage!r} was requested"
            )
        ws = t["dx_ws"] if k > 1 else t["f32_dummy"]
        values.update(
            A=dz_chunk,
            B=t["W"],
            C=t["dx_acc"][row0:stop],
            STATS_OUT=t["stats"],
            WS=ws,
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
                ws_slab=int(ws.stride(0)) if k > 1 else 0,
            ),
        )
    elif stage == "slab_sum":
        # the fused dX finalize's slab reduction of the K-sliced dX GEMM: one kernel over the chunk's rows of dX_acc
        k = plan.dx_slices_of(index)
        if not plan.dx_finalize or k <= 1:
            raise ValueError(
                f"slab_sum belongs to a K-sliced dX GEMM of a plan with dx_finalize (chunk {index}: {k} slice(s))"
            )
        ws = t["dx_ws"]
        values.update(
            dx=t["dx_acc"][row0:stop],
            ws=ws,
            ws_slab=int(ws.stride(0)),
            num_vecs=int(rows_c * p.hidden // g.cast_vec),
            n_slabs=int(k - 1),
        )
    elif (
        stage == "dx_reduce"
    ):  # host-side slab reduction of the K-sliced dX GEMM (no kernel of the program)
        values.update(
            acc=t["dx_acc"][row0:stop], ws=t["dx_ws"], k_slices=plan.dx_slices_of(index)
        )
    elif base == "gemm_dw_acc":
        want = plan.dw_acc_stage(index)
        if stage != want:
            raise ValueError(
                f"chunk {index} plans the weight-gradient accumulate {want!r}; stage {stage!r} was requested"
            )
        v = plan.variants_of(index)
        values.update(
            A=dz_chunk,
            B=x_chunk,
            C=t["dw_acc"],
            STATS_OUT=t["stats"],
            WS=t["f32_dummy"],
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
            term=_term_rows(t, plan, index),
        )
    elif stage == "loss_reduce":
        values.update(
            term=_term_rows(t, plan, index), loss_acc=t["loss_acc"], loss_out=t["loss"]
        )
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


def _term_rows(t: dict[str, Any], plan: Plan, index) -> torch.Tensor:
    """The ``term`` rows chunk ``index``'s ``row_finalize`` writes and its ``loss_reduce`` reads: the chunk-local
    ``term`` buffer -- or, in a forward that keeps its loss terms (``t["keep_terms"]``, the exact loss of a zero-weight
    split, :func:`_zero_weight_exact_loss`), for every chunk after the first the compact ``loss_weights`` rows of the
    chunk before it (``[row0 - C, row0 - C + rows_c)``): the gather filled them before the loop and no launch reads
    them again once that chunk's ``row_finalize`` ran, so the terms of chunk ``c >= 1`` outlive the loop at
    ``loss_weights[(c - 1) C + r]`` and chunk 0's in ``term`` without another buffer.  Both kernels index their
    ``term`` argument chunk-locally (``term[r]``), so only the base moves."""
    if not t.get("keep_terms") or index == 0:
        return t["term"]
    row0, rows_c = plan.chunks[index]
    shift = int(row0) - int(plan.chunks[0][1])
    return t["loss_weights"][shift : shift + int(rows_c)]


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
    elif base == "gemm_dx":
        values.update(
            A=logits[:rows_c],
            B=t["W"],
            C=t["dx_acc"][row0:stop],
            STATS_OUT=t["stats"],
            WS=t["dx_ws"] if plan.dx_slices_of(index) > 1 else t["f32_dummy"],
        )
    elif stage == "slab_sum":
        values.update(dx=t["dx_acc"][row0:stop], ws=t["dx_ws"])
    elif base == "gemm_dw_acc":
        values.update(
            A=logits[:rows_c],
            B=x_chunk,
            C=t["dw_acc"],
            STATS_OUT=t["stats"],
            WS=t["f32_dummy"],
        )
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
            term=_term_rows(t, plan, index),
        )
    elif stage == "loss_reduce":
        values.update(
            term=_term_rows(t, plan, index), loss_acc=t["loss_acc"], loss_out=t["loss"]
        )
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


def _reads_labels_temperature(stage: str) -> bool:
    """Whether the host values of ``stage`` (:func:`stage_values`) include the call's labels."""
    base = (
        base_stage(stage)
        if stage in GEMM_STAGES or stage in TEMPERATURE_STAGES
        else stage
    )
    return "labels" in STAGE_TENSORS.get(base, ())


def _dx_keys(plan: Plan, index: int) -> list:
    """The dX GEMM of chunk ``index`` plus, when the plan slices it, the slab reduction: the ``slab_sum`` kernel of the
    fused dX finalize or the host's ``add_`` chain (``dx_reduce``)."""
    k = plan.dx_slices_of(index)
    if k <= 1:
        return [(plan.dx_stage_of(index), index)]
    reduce = "slab_sum" if plan.dx_finalize else "dx_reduce"
    return [(plan.dx_stage_of(index), index), (reduce, index)]


def recompute_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys of the log-probability backward (recompute the logits, then the gradient GEMMs).  A kept-logits
    plan (:class:`_KeptPlan`) recomputes nothing: ``row_grad`` forms each chunk's ``dz`` over its kept logits
    (the chunk's ``X`` rows are gathered for the weight-gradient GEMM only)."""
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
            # the recompute stores ``dz`` itself (its ``_dcnt`` form under the zero-gradient row skip)
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


def _recompute_keys_temperature(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys of the log-probability backward (recompute the logits, then the gradient GEMMs)."""
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
            # ``0 - acc`` as the kernel forms it (``+0`` for a zero sum of either sign; a unary minus would give ``-0``)
            neg = torch.zeros_like(acc).sub_(acc)
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

    gemm_dx_s = gemm_dx  # the reference path never slices K (one product per chunk)

    @staticmethod
    def dx_reduce(values: dict[str, Any]) -> None:
        _dx_reduce(values)

    @staticmethod
    def slab_sum(values: dict[str, Any]) -> None:
        # the fused dX finalize's slab reduction: the same fixed-order FP32 adds as ``dx_reduce``
        dx, ws = values["dx"], values["ws"]
        for slab in range(int(values["n_slabs"])):
            dx.add_(ws[slab, : dx.shape[0]])

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
class LmHeadDetLossRunner:
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
    :func:`prepare_lm_head_det_loss`); it may be reassigned between steps (a CUDA graph records the mode it was captured
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
    # built only to template a remembered binding on a hidden valid-row count's chunk-0 buffers
    # (``prepare_lm_head_det_loss(_chunk_views=...)``): those buffers hold the count's chunk 0, so it never launches
    _template_only: bool = False

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
        if self._template_only:
            raise RuntimeError(
                "this runner only templates a remembered binding (its chunk buffers are a hidden valid-row count's); "
                "it never launches"
            )
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
        """Loss entry: ``(loss [1], logp [T])``; log-probability entry: ``(logp, lse)``."""
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
        tensor; ``grad`` must be ``None`` (the cast scale is 1).
        """
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


class _TemperatureRunner(LmHeadDetLossRunner):
    """A prepared runner of a plan with the temperature forms (:attr:`Plan.temperature`): its chunk loop takes the
    side-stream placement of the temperature forms (:func:`_run_keys_temperature`)."""

    def _run(self, keys: tuple) -> None:
        if self.backend != "cake" or self._template_only:
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


class _KeptLossRunner(LmHeadDetLossRunner):
    """The prepared runner of a kept-logits plan (:class:`_KeptPlan`): its kept views are bound per call, so only the
    kept-logits entry points launch it (through :class:`_KeptBinding`)."""

    def forward(self):
        raise NotImplementedError(_KEPT_RUNNER)

    def backward(self, grad: Optional[torch.Tensor] = None):
        raise NotImplementedError(_KEPT_RUNNER)


_KEPT_RUNNER = (
    "a kept-logits plan (logprob_keep) binds its kept views per call: launch it through the eager entry points "
    "(forward_logprob_keep / backward_logprob_keep / cake_lm_head_det_train_logprob_keep)"
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
        if (
            stage in HOST_STAGES
        ):  # host-side step (slab reduction, row gather), no kernel
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


def prepare_lm_head_det_loss(
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
    logprob_keep: bool = False,
    dcnt: Optional[torch.Tensor] = None,
    dx_slices_cap: Optional[int] = None,
    _chunk_views: Optional["_HiddenCount"] = None,
) -> LmHeadDetLossRunner:
    """Validate one binding and prepare its launches.

    Missing outputs, accumulators and the workspace are allocated here (the
    only allocations of the ``cake`` backend).  Pass ``workspace_buffer`` of
    :func:`lm_head_det_loss_workspace_size` bytes to reuse storage across steps.
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

    ``dx_finalize`` (default :func:`dx_finalize_default`): a K-sliced dX GEMM's
    slabs are added into ``dx_acc`` by the ``slab_sum`` kernel instead of the
    host's ``add_`` chain (bitwise the same accumulator).  ``dx_cast=False``
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
    ``NotImplementedError``).

    ``dcnt`` (log-probability entry, compacted rows ordered by
    :func:`zero_dlogp_partition`): the int32 per-chunk counts of live rows; the
    plan then runs the ``_dcnt`` forms (:attr:`Plan.dcnt`) where the record
    registers every one it needs, and the forms without the count otherwise
    (every row computed).  The counted plan's ``dX_acc`` rows past a chunk's
    count are not defined: only the fused dX finalize with the partition's
    scan (:func:`finalize_dx`) reads such an accumulator.

    ``dx_slices_cap`` (loss entry; the zero-weight split of :func:`forward_loss`):
    the plan's rule-picked dX slice count per chunk stays at or below it
    (:func:`make_plan`), so the K-slice workspace never exceeds the one of the
    call without the split.

    ``_chunk_views`` (private; the call's hidden valid-row count, ``cake``
    backend): a runner built only to template a remembered binding
    (:meth:`_Binding.from_runner`) for a call that missed the binding cache --
    its logits, statistics and gather regions are views of the count's chunk-0
    buffers (:data:`CHUNK_VIEW_REGIONS`) and the other regions one smaller
    allocation (:func:`_chunk_view_layout`), so preparing it beside those
    buffers allocates no second chunk buffer; it never launches.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if _chunk_views is not None and (backend != "cake" or workspace_buffer is not None):
        raise ValueError(
            "_chunk_views templates a cake-backend runner on its own workspace"
        )
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
    if backend == "cake":
        if not X.is_cuda:
            raise ValueError("the cake backend needs CUDA tensors")
        module_name, record = record_for(device)
        record_abi(record)
    geometry = Geometry.from_record(record)
    problem = validate_lm_head_det_inputs(
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
        logprob_keep=keep,
        dx_slices_cap=None if dx_slices_cap is None else int(dx_slices_cap),
    )
    plan = plan_of(dcnt=dcnt is not None)
    if plan.dcnt and any(s not in stages for s in plan.stages):
        # a record without the counted TMA cast: the counted plan with the shipped cast's counted form
        plan = plan_of(dcnt=True, counted_tma_cast=False)
    if plan.dcnt and any(s not in stages for s in plan.stages):
        # a record without the counted forms: the plan without them, every row computed
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
    tma_bytes = _record_tma_bytes(record, stages) if record is not None else 0
    scratch_bytes = _record_scratch_bytes(record, stages) if record is not None else 0
    layout = workspace_layout(
        rows_loop,
        V,
        C,
        stats_tile=geometry.stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_bytes,
        scratch_bytes=scratch_bytes,
        dx_ws_slabs=plan.dx_ws_slabs,
        hidden=H,
        compact=plan.compact,
    )
    # a template on the hidden valid-row count's chunk-0 buffers: the other regions in one smaller allocation
    carved = layout if _chunk_views is None else _chunk_view_layout(layout)
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            carved["total"], dtype=torch.uint8, device=device
        )
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < carved["total"]:
        raise ValueError(
            f"workspace_buffer needs {carved['total']} bytes, got {flat.numel()}"
        )

    def carve(name, dtype, shape):
        if _chunk_views is not None and name in CHUNK_VIEW_REGIONS:
            return _chunk_view(_chunk_views, name, shape)
        return _carve(flat, carved, name, dtype, shape)

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
    t["logits"] = carve("logits", torch.bfloat16, (rows, V))
    t["stats"] = carve("stats", torch.float32, (rows, plan.num_tiles, 2))
    t["d"] = carve("d", torch.float32, (rows,))
    t["term"] = carve("term", torch.float32, (rows,))
    t["loss_acc"] = carve("loss_acc", torch.float64, (1,))
    t["grad_scale"] = carve("grad_scale", torch.float32, (1,))
    t["unit_scale"] = torch.ones(
        (1,), dtype=torch.float32, device=device
    )  # the log-probability entry's cast scale
    t["f32_dummy"] = torch.zeros(
        (16,), dtype=torch.float32, device=device
    )  # unused ``WS`` of the unsliced GEMMs
    if plan.dx_ws_slabs:
        t["dx_ws"] = carve("dx_ws", torch.float32, (plan.dx_ws_slabs, rows, H))
    if plan.compact:
        t["row_index"] = row_index
        t["x_c"] = carve(
            "x_c", torch.bfloat16, (rows, H)
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
            torch.float32 if _dw_cast_f32(plan.dw_cast_stage) else torch.bfloat16,
        )
    if layout.get("workspace"):
        t["workspace"] = carve("workspace", torch.uint8, (layout["workspace"][1],))
    if layout.get("tma_descriptor_workspace"):
        t["tma_descriptor_workspace"] = carve(
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
        dx_ws_slabs=plan.dx_ws_slabs,
        valid_rows=plan.valid_rows,
        fuse_dw_cast=plan.fuse_dw_cast,
        dx_finalize=plan.dx_finalize,
    )
    if plan.logprob_keep:
        memory = kept_memory_report(memory, plan)
    if device.type == "cuda":
        device_index = int(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    else:
        device_index = 0
    return (_KeptLossRunner if plan.logprob_keep else LmHeadDetLossRunner)(
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
        _template_only=_chunk_views is not None,
    )


def prepare_lm_head_det_loss_temperature(
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
    temperature: Optional[torch.Tensor] = None,
    dcnt: Optional[torch.Tensor] = None,
    _chunk_views: Optional["_HiddenCount"] = None,
) -> LmHeadDetLossRunner:
    """Validate one binding and prepare its launches.

    Missing outputs, accumulators and the workspace are allocated here (the
    only allocations of the ``cake`` backend).  Pass ``workspace_buffer`` of
    :func:`lm_head_det_loss_workspace_size` bytes to reuse storage across steps.
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

    ``dx_finalize`` (default :func:`dx_finalize_default`): a K-sliced dX GEMM's
    slabs are added into ``dx_acc`` by the ``slab_sum`` kernel instead of the
    host's ``add_`` chain (bitwise the same accumulator).  ``dx_cast=False``
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
    counts of live rows, as in :func:`prepare_lm_head_det_loss`; with a temperature the plan of
    :func:`make_plan_temperature` whose forms the record registers: the counted ``dz`` temperature form
    (:func:`zero_dlogp_temperature_dz_counted`), else the mixed plan (the uncounted ``dz`` temperature form), each
    with the counted dW forms, and the forms without the count otherwise.

    ``_chunk_views`` (private): a template runner on the hidden valid-row count's chunk-0 buffers, as in
    :func:`prepare_lm_head_det_loss`; it never launches.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if _chunk_views is not None and (backend != "cake" or workspace_buffer is not None):
        raise ValueError(
            "_chunk_views templates a cake-backend runner on its own workspace"
        )
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
    if backend == "cake":
        if not X.is_cuda:
            raise ValueError("the cake backend needs CUDA tensors")
        module_name, record = record_for(device)
        record_abi(record)
    geometry = Geometry.from_record(record)
    problem = validate_lm_head_det_inputs(
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
    )
    # under the skip, the first plan whose stages the record registers: the counted temperature dz arm
    # (:func:`zero_dlogp_temperature_dz_counted`), then the uncounted one (the mixed plan), each with the counted TMA
    # cast, then with the shipped cast's counted form; else the plan without the counted forms, every row computed
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
        for dz_form in dz_forms:
            for cast_form in ({}, {"counted_tma_cast": False}):
                plan = plan_of(dcnt=True, **dz_form, **cast_form)
                if not plan.dcnt or all(s in stages for s in plan.stages):
                    break
            if not plan.dcnt or all(s in stages for s in plan.stages):
                break
    if plan.dcnt and any(s not in stages for s in plan.stages):
        # a record without the counted forms: the plan without them, every row computed
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
    tma_bytes = _record_tma_bytes(record, stages) if record is not None else 0
    scratch_bytes = _record_scratch_bytes(record, stages) if record is not None else 0
    layout = workspace_layout(
        rows_loop,
        V,
        C,
        stats_tile=geometry.stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_bytes,
        scratch_bytes=scratch_bytes,
        dx_ws_slabs=plan.dx_ws_slabs,
        hidden=H,
        compact=plan.compact,
    )
    # a template on the hidden valid-row count's chunk-0 buffers: the other regions in one smaller allocation
    carved = layout if _chunk_views is None else _chunk_view_layout(layout)
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            carved["total"], dtype=torch.uint8, device=device
        )
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < carved["total"]:
        raise ValueError(
            f"workspace_buffer needs {carved['total']} bytes, got {flat.numel()}"
        )

    def carve(name, dtype, shape):
        if _chunk_views is not None and name in CHUNK_VIEW_REGIONS:
            return _chunk_view(_chunk_views, name, shape)
        return _carve(flat, carved, name, dtype, shape)

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
    t["logits"] = carve("logits", torch.bfloat16, (rows, V))
    t["stats"] = carve("stats", torch.float32, (rows, plan.num_tiles, 2))
    t["d"] = carve("d", torch.float32, (rows,))
    t["term"] = carve("term", torch.float32, (rows,))
    t["loss_acc"] = carve("loss_acc", torch.float64, (1,))
    t["grad_scale"] = carve("grad_scale", torch.float32, (1,))
    t["unit_scale"] = torch.ones(
        (1,), dtype=torch.float32, device=device
    )  # the log-probability entry's cast scale
    t["f32_dummy"] = torch.zeros(
        (16,), dtype=torch.float32, device=device
    )  # unused ``WS`` of the unsliced GEMMs
    if plan.dx_ws_slabs:
        t["dx_ws"] = carve("dx_ws", torch.float32, (plan.dx_ws_slabs, rows, H))
    if plan.compact:
        t["row_index"] = row_index
        t["x_c"] = carve(
            "x_c", torch.bfloat16, (rows, H)
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
            torch.float32 if _dw_cast_f32(plan.dw_cast_stage) else torch.bfloat16,
        )
    if layout.get("workspace"):
        t["workspace"] = carve("workspace", torch.uint8, (layout["workspace"][1],))
    if layout.get("tma_descriptor_workspace"):
        t["tma_descriptor_workspace"] = carve(
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
        dx_ws_slabs=plan.dx_ws_slabs,
        valid_rows=plan.valid_rows,
        fuse_dw_cast=plan.fuse_dw_cast,
        dx_finalize=plan.dx_finalize,
        temperature=problem.temperature,
    )
    if device.type == "cuda":
        device_index = int(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    else:
        device_index = 0
    runner = _TemperatureRunner if plan.temperature else LmHeadDetLossRunner
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
        _template_only=_chunk_views is not None,
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
    """``(shape, stride, dtype)``: the key of a per-step operand whose address no plan reads (the remembered launches
    re-bind it on every call)."""
    return None if t is None else (tuple(t.shape), tuple(t.stride()), t.dtype)


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
) -> tuple:
    """Cache key of a forward binding: ``(data_ptr, shape, stride, dtype)`` of every
    input plus every option that shapes the argument plans; ``valid_rows`` is the
    compacted row count (a label-dependent fact of the plan; ``None`` = uncompacted);
    ``fuse_dw_cast`` the resolved weight-gradient form; ``gemm_tuning`` the explicit
    GEMM knobs (``None`` = none, :func:`gemm_tuning_default`);
    ``dx_finalize`` the resolved dX finalize form (the slab reduction's launches differ);
    ``hidden_count`` whether chunk 0's gather and logits GEMM ran before the plan
    (the remembered sequence starts at chunk 0's row kernels); ``logprob_lbl``
    the resolved log-probability forward's logits form (:func:`logprob_lbl_default`)."""
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
    zero_dlogp=False,
) -> tuple:
    """Cache key of a log-probability backward binding (as :func:`forward_binding_key`); ``logprob_dz`` the resolved
    recompute form (:func:`logprob_dz_default`), ``zero_dlogp`` whether the call is eligible for the zero-gradient row
    skip (:func:`_zero_dlogp_skip`; the remembered plan tells whether it runs its counted forms, so a remembered
    binding replays the plan the call took).  ``lse`` and ``dlogp`` are keyed by layout alone (:func:`_layout`):
    the forward's statistic and the incoming gradient are fresh tensors on every step, so their addresses would make
    every step whose allocations moved a miss (a new runner, its allocations and validation), while the remembered
    launches re-bind both per call."""
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
# the caching allocator (their sizes are part of the plan; the allocator reuses the blocks).
_OWNED_VALUES = ("workspace", "tma_descriptor_workspace")


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

    @classmethod
    def from_runner(
        cls, runner: LmHeadDetLossRunner, hidden: Optional[_HiddenCount] = None
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
            forward_late=tuple(key for key in forward_order if key not in early),
            owned=owned,
            owned_bytes=sum(v.numel() * v.element_size() for v in owned.values()),
            layout=runner.layout,
            memory=memory,
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
        stats: bool = True,
        arena: Optional[torch.Tensor] = None,
    ) -> None:
        """Per-call temporaries from the caching allocator into ``t`` (the runner's regions, minus the owned ones);
        with ``hidden`` the chunk buffers are the prologue's (chunk 0's logits / statistics / gathered rows already in
        them, ``min(chunk, T)`` rows) -- the entry points check that a hidden-count plan gets its prologue.
        ``chunk_buffers=False``: the chunk buffers are already bound (:meth:`_chunk_buffers`, the early launch of
        :meth:`forward`) and the rest is allocated.  ``stats=False`` (the backward of a plan under the zero-gradient
        row skip, :meth:`backward_logprob`): no row-statistics buffer; ``stats`` is bound to the constant ``f32_dummy``
        cell -- no launch of that backward writes or reads its ``STATS_OUT``: the ``dz`` arm's forms store no
        statistics, the counted weight-gradient accumulate takes no ``STATS_OUT`` and the uncounted dX GEMM's epilogue
        never touches it (the fused cast reads the separate scale cell).  ``arena`` (:func:`_arena_views`): the chunk
        buffers are views of its storage when they fit."""
        p = self.plan.problem
        rows = self.layout["logits"][1] // (p.vocab * 2)
        carved = None
        if chunk_buffers and hidden is None:
            carved = _arena_views(arena, self._chunk_specs(rows, stats=stats))
        if chunk_buffers and hidden is not None:
            t["logits"], t["stats"] = hidden.logits, hidden.stats_buf
        elif carved is not None:
            t.update(carved)
        elif chunk_buffers:
            t["logits"] = torch.empty(
                (rows, p.vocab), dtype=torch.bfloat16, device=self.device
            )
            if stats:
                t["stats"] = torch.empty(
                    (rows, self.plan.num_tiles, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
        t["d"] = torch.empty((rows,), dtype=torch.float32, device=self.device)
        t["term"] = torch.empty((rows,), dtype=torch.float32, device=self.device)
        t["loss_acc"] = torch.empty((1,), dtype=torch.float64, device=self.device)
        t["loss"] = torch.empty((1,), dtype=torch.float32, device=self.device)
        t.update(
            _device_constants(self.device_index)
        )  # constant cells: no per-call fill launches
        if "stats" not in t:
            t["stats"] = t["f32_dummy"]  # ``stats=False``: never written or read
        if self.plan.dx_ws_slabs:
            t["dx_ws"] = torch.empty(
                (self.plan.dx_ws_slabs, rows, p.hidden),
                dtype=torch.float32,
                device=self.device,
            )
        if self.plan.compact and chunk_buffers and carved is None:
            t["x_c"] = (
                hidden.x_c
                if hidden is not None
                else torch.empty(
                    (rows, p.hidden), dtype=torch.bfloat16, device=self.device
                )
            )
        for name, tensor in self.owned.items():
            t[name] = tensor

    def _chunk_specs(
        self, rows: int, *, stats: bool = True
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """``name -> (shape, dtype)`` of the chunk buffers: logits, statistics (``stats``) and, compacted, gathered rows."""
        p = self.plan.problem
        specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "logits": ((rows, p.vocab), torch.bfloat16)
        }
        if stats:
            specs["stats"] = ((rows, self.plan.num_tiles, 2), torch.float32)
        if self.plan.compact:
            specs["x_c"] = ((rows, p.hidden), torch.bfloat16)
        return specs

    def _chunk_buffers(
        self, t: dict[str, Any], arena: Optional[torch.Tensor] = None
    ) -> None:
        """What chunk 0's row gather and logits GEMM bind (the early launch of :meth:`forward`): the chunk's logits,
        statistics and gather buffers of :meth:`_scratch` (the same allocations, ahead of the others; views of
        ``arena``'s storage when they fit, :func:`_arena_views`), the constant cells and the owned regions."""
        p = self.plan.problem
        rows = self.layout["logits"][1] // (p.vocab * 2)
        specs = self._chunk_specs(rows)
        carved = _arena_views(arena, specs)
        if carved is None:
            carved = {
                name: torch.empty(shape, dtype=dtype, device=self.device)
                for name, (shape, dtype) in specs.items()
            }
        t.update(carved)
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

    def _launch(self, keys: tuple, t: dict[str, Any], dw_stream: str = "auto") -> None:
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

        _run_keys(self.plan, self.device_index, keys, run_key, dw_stream)

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
        capture: Optional[dict] = None,
        arena: Optional[torch.Tensor] = None,
        dw_acc: Optional[torch.Tensor] = None,
    ):
        """One forward through the remembered launches; ``dw_stream`` is the call's dW side-stream mode
        (:func:`dw_stream_mode`), a launch placement outside the binding.  ``capture`` (a loss forward of a zero-weight
        split with the exact loss, :func:`_zero_weight_exact_loss`): the chunk loop keeps its loss terms
        (:func:`_term_rows`) and ``capture`` receives the two buffers holding them, ``term`` and ``loss_weights``.
        ``arena`` (the zero-weight rows' logp-only pass ahead of the chunk loop, :func:`_arena_views`): the chunk buffers are
        views of its storage when they fit; ``dw_acc``: the FP32 ``[V, H]`` accumulator to use (that arena) instead of
        a new one.

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
        if capture is not None:
            if not (p.entry == "loss" and plan.compact and infer_logp is not None):
                raise ValueError(
                    "only a compacted policy loss forward keeps its loss terms"
                )
            t["keep_terms"] = True
        if early:
            self._compact(t, row_index)
            self._chunk_buffers(t, arena)
            if self.forward_early:
                self._launch(self.forward_early, t, dw_stream)
            t["labels"] = _prepare_labels(labels, plan.geometry, row_index)
            if self.forward_early_labels:
                self._launch(self.forward_early_labels, t, dw_stream)
            self._scratch(t, chunk_buffers=False)
        else:
            t["labels"] = _prepare_labels(labels, plan.geometry, row_index)
            self._compact(t, row_index)
            self._scratch(t, hidden, arena=arena)
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
                t["dw_acc"] = (
                    dw_acc
                    if dw_acc is not None
                    else torch.empty((V, H), dtype=torch.float32, device=self.device)
                )
        self._launch(self.forward_late if early else self.forward_order, t, dw_stream)
        if capture is not None:
            capture.update(term=t["term"], loss_weights=t["loss_weights"])
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
        """One log-probability backward through the remembered launches.  A plan under the zero-gradient row skip
        (:attr:`Plan.dcnt`) takes the call's per-chunk counts ``dcnt`` (:func:`zero_dlogp_partition`; the rows of
        ``row_index``, ``lse`` and the temperature rows in its order) and binds no row-statistics buffer
        (:meth:`_scratch` ``stats=False``)."""
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
            _check_output(dcnt, "dcnt", (plan.num_chunks,), torch.int32)
            t["dcnt"] = dcnt
        self._scratch(t, hidden, stats=not plan.dcnt)
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
        cls, runner: LmHeadDetLossRunner, hidden: Optional[_HiddenCount] = None
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
            forward_late=tuple(key for key in forward_order if key not in early),
            owned=owned,
            owned_bytes=sum(v.numel() * v.element_size() for v in owned.values()),
            layout=runner.layout,
            memory=memory,
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
        arena: Optional[torch.Tensor] = None,
        dw_acc: Optional[torch.Tensor] = None,
    ):
        """:meth:`_Binding.forward` with the call's compact temperature rows."""
        return self._with_rows(
            temperature,
            functools.partial(super().forward, arena=arena, dw_acc=dw_acc),
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
        (:attr:`Plan.dcnt`), the per-chunk counts ``dcnt``."""
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

    def _launch(self, keys: tuple, t: dict[str, Any], dw_stream: str = "auto") -> None:
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

        _run_keys_temperature(self.plan, self.device_index, keys, run_key, dw_stream)


class _KeptBinding(_Binding):
    """A remembered binding of a kept-logits plan (:class:`_KeptPlan`): the templated launches of :class:`_Binding`,
    with every launch of chunk ``i`` bound to the chunk's kept view (one BF16 ``[rows_c, V]`` view per chunk,
    :func:`_kept_views`) where the default plan binds the chunk buffer.  Launched only by the kept-logits entry points
    (:meth:`forward_keep`, :meth:`backward_logprob_keep`)."""

    def _launch(self, keys: tuple, t: dict[str, Any], dw_stream: str = "auto") -> None:
        kept = t.get("logits_keep")
        if kept is None:
            # chunk 0's early launches, ahead of the kept views: chunk 0's view is the chunk buffer's first rows
            super()._launch(keys, t, dw_stream)
            return
        plan, launches = self.plan, self.launches

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

        _run_keys(self.plan, self.device_index, keys, run_key, dw_stream)

    def forward_keep(
        self, X, W, labels, row_index=None, hidden=None, dw_stream: str = "auto"
    ):
        """:meth:`_Binding.forward` of the log-probability entry that keeps the logits: ``(logp, lse, kept)``, ``kept``
        one BF16 ``[rows_c, V]`` view per chunk (:func:`_kept_views`; chunk 0's in the chunk buffer)."""
        p, plan = self.plan.problem, self.plan
        T = plan.rows
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
        kept = t["logits_keep"] = _kept_views(plan, t["logits"], self.device)
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
    ):
        """:meth:`_Binding.backward_logprob` over the forward's kept views ``logits`` (:func:`check_kept_logits`):
        ``row_grad`` overwrites each with the chunk's ``dz``; no chunk logits buffer is allocated."""
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
        t["stats"] = torch.empty(
            (rows, plan.num_tiles, 2), dtype=torch.float32, device=self.device
        )
        if plan.compact:
            t["x_c"] = torch.empty((rows, H), dtype=torch.bfloat16, device=self.device)
        self._scratch(t, chunk_buffers=False)
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
        self._launch(self.backward_order, t, dw_stream)
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

    Threads share the cache without a lock: each ordered-dict operation is atomic, and a step that finds its key or
    entry gone -- another thread's eviction between a lookup's ``get`` and its recency update, or between an
    eviction's size check and its ``popitem`` -- skips that step (a looked-up binding stays valid for its call; an
    eviction that finds the cache already within its capacity stops).  The hit / miss counters are statistics."""

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


# Enabled, 64 bindings.  ``flashinfer.cake_lm_head_det_train`` applies its environment variables through
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


@dataclass
class ForwardResult(_TemperatureResultState):
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
    finite ``g`` beyond the BF16 range still gives a signed zero."""
    if out_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("the cast produces bfloat16 or float32")
    zero = torch.zeros((), dtype=torch.float32, device=acc.device)
    if grad is not None:
        zero = zero * grad.detach().reshape(()).to(
            device=acc.device, dtype=torch.float32
        )
    return torch.empty(acc.shape, dtype=out_dtype, device=acc.device).copy_(zero)


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
) -> Optional[_HiddenCount]:
    """The hidden valid-row count prologue of a compacted ``cake`` call when the switch (``hidden_count``, resolved by
    the caller) is on and the call is eligible (:func:`hidden_count_eligible`); ``None`` = the shipped
    :func:`valid_rows` path.  ``loss_weights``: the zero-weight split (:func:`gradient_index`)."""
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
    :func:`prepare_lm_head_det_loss` and launches through the templated binding it
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
    once here) adds a sliced dX GEMM's slabs with the ``slab_sum`` kernel and, on
    a compacted call, keeps the valid-row mask (``row_valid``) for
    :func:`backward_loss`'s one-pass scatter of ``dX``.  ``dw_stream`` (default
    ``None`` = ``"auto"``, :func:`dw_stream_mode`) places each chunk's
    weight-gradient accumulate on the per-device side stream (``True`` every
    multi-chunk call, ``False`` never, ``"auto"`` calls of
    :data:`DW_STREAM_MIN_CHUNKS` chunks or more).  ``hidden_count`` (default
    :func:`hidden_count_default`) runs a compacted call's valid-row count on the
    device.  ``gemm_tuning`` (default :func:`gemm_tuning_default`) pins GEMM
    knobs explicitly; the result carries it for the deferred weight gradient.

    Zero-weight rows (a compacted policy call, :func:`zero_weight_split`): the chunk loop runs over the gradient rows
    ``(labels >= 0) & (loss_weights != 0)`` only (``row_index``, ``row_valid``, ``dx_acc`` and the deferred operands
    are theirs); the zero-weight rows contribute exact zeros to ``dX`` and ``dW``.  The loss follows
    :func:`zero_weight_loss_mode` (``"exact"``: the bits of the call without the split,
    :func:`_zero_weight_exact_loss`).  ``return_logp`` (default ``True``): their ``logp`` comes from a logp-only pass
    of the log-probability forward (:func:`_zero_weight_logp`); ``False`` skips that pass and leaves their ``logp`` at
    0 (the loss autograd entry without ``return_logp``).  Both row counts come back in the call's one count readback.
    The memory rule (:func:`zero_weight_plan`, host-only on those counts): the split's chunk loop runs at most the dX K
    slices per chunk of the call without the split, and a call whose predicted split peak
    (:func:`zero_weight_memory`) exceeds its predicted peak without the split runs without it.
    A call whose valid rows all have zero weight launches no gradient GEMM and returns zero accumulators, as a call
    without valid rows.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_det_inputs(
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
        zero_index = hc.zero_index() if split else None
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
    dx_cap: Optional[int] = None
    zero_order = zero_arena = False
    if zero_rows:
        # the memory rule, on the host from the two counts: the split's chunk loop runs at most the dX K slices per
        # chunk of the call without the split, and the call splits only when the split's predicted peak stays within
        # that call's (else it runs that call: its valid-row index from the counts, no further synchronization; the
        # hidden count's queued chunk-0 work is dropped)
        grad_rows = 0 if row_index is None else int(row_index.numel())
        zplan = zero_weight_plan(
            problem,
            X.device,
            grad_rows=grad_rows,
            zero_rows=zero_rows,
            backend=backend,
            need_dx=need_dx,
            need_dw=need_dw,
            fuse_dw_cast=fuse,
            dx_finalize=fused_dx,
            hidden_count=hc is not None,
            return_logp=bool(return_logp),
            exact_loss=zero_weight_loss_mode() == "exact",
            tuning=tuning,
        )
        if zplan["split"]:
            dx_cap, zero_order = zplan["dx_slices_cap"], bool(zplan["zero_first"])
            zero_arena = bool(zplan["pass_arena"])
        else:
            hc = None
            row_index, row_valid = _valid_rows_of_count(
                labels, grad_rows + zero_rows, mask=fused_dx
            )
            zero_index, zero_rows = None, 0
    exact = bool(zero_rows) and zero_weight_loss_mode() == "exact"
    # Where the zero-weight rows' logp-only pass runs (:func:`zero_weight_memory`'s ``zero_first``): ahead of the chunk
    # loop when its chunk buffers go into the loop's FP32 dW accumulator (``pass_arena``, below) or the loop defers its
    # last chunk; else after the loop / in place of it (no chunk loop: every valid row has zero weight), in the loop's
    # freed chunk buffers, with ``hc`` released -- the two passes' chunk buffers never coexist and the pass adds no
    # allocator segment.  With the hidden count and ``fuse`` the pass runs where the model predicts the lower peak: ahead
    # of the loop next to chunk 0's buffers (``chunk0_alive``, counted in the memory report's peak) or after it next to
    # the accumulators and the deferred chunk's ``dz`` rows.
    zero_pass = bool(zero_rows and return_logp)
    zero_first = zero_pass and zero_order
    chunk0_alive = zero_first and hc is not None
    no_loop = (
        row_index is not None and row_index.numel() == 0
    )  # every row ignored (or of zero weight)
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
    # the exact loss of a zero-weight split: the chunk loop keeps its loss terms and hands their two buffers over
    terms: Optional[dict] = {} if exact else None
    cache = BINDING_CACHE
    key = None
    binding = None
    # the pass arena below (the loop's FP32 weight-gradient accumulator, allocated ahead of the logp-only pass)
    with_arena = bool(zero_arena and zero_first and backend == "cake")
    if backend == "cake" and not no_loop:
        if cache.enabled:
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
                **common,
            )
            if dx_cap is not None:  # the split's capped dX slice counts
                key += ("dx_slices_cap", int(dx_cap))
            binding = cache.lookup(key)
        if binding is None and with_arena:
            # a call that misses the cache prepares its binding (below) before the arena exists, so the runner's
            # workspace and the arena never coexist
            runner = prepare_lm_head_det_loss(
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
                dx_slices_cap=dx_cap,
                **common,
            )
            binding = _Binding.from_runner(runner, hc)
            del runner
            if key is not None:
                cache.remember(key, binding)
    # Ahead of a loop that keeps the FP32 weight-gradient accumulator (``pass_arena``, :func:`zero_weight_memory`): the
    # accumulator is allocated first and the pass's chunk buffers are views of it (:func:`_arena_views`; the loop writes
    # it before reading it), so the reserved peak stays within the plain call's.
    arena = (
        torch.empty(
            (problem.vocab, problem.hidden), dtype=torch.float32, device=X.device
        )
        if with_arena
        else None
    )
    zero_logp = (
        _zero_weight_logp(
            problem,
            X,
            W,
            labels,
            zero_index,
            backend=backend,
            gemm_tuning=tuning,
            arena=arena,
        )
        if zero_first
        else None
    )
    if no_loop:
        hc = None
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
        if exact:
            # the plain path's reduction of the zero-weight rows' terms (their weights): its bits on either backend
            result.loss = _zero_weight_exact_loss(
                problem, None, row_index, zero_index, loss_weights, backend=backend
            ).reshape(())
        if zero_pass and not zero_first:
            zero_logp = _zero_weight_logp(
                problem, X, W, labels, zero_index, backend=backend, gemm_tuning=tuning
            )
        return _with_zero_weight_rows(
            result, zero_index, zero_logp, chunk0_alive=chunk0_alive, exact=exact
        )
    if binding is not None:
        memory = binding.memory
    else:
        runner = prepare_lm_head_det_loss(
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
            dx_slices_cap=dx_cap,
            **common,
        )
        if backend == "cake":
            # The first call launches through the templated binding too: its per-call temporaries come from the caching
            # allocator, so the deferred ``dz`` rows keep the chunk buffer alive and nothing else -- a view into the
            # runner's single workspace allocation would pin the whole workspace until the backward.  The runner (and
            # its workspace) is dropped here; only the binding survives.
            binding = _Binding.from_runner(runner, hc)
            del runner
            if key is not None:
                cache.remember(key, binding)
            memory = binding.memory
        else:
            if terms is not None:
                _keep_runner_terms(runner)
            loss, logp = runner.forward()
            dx_acc, dw_acc = runner.dx_acc, runner.dw_acc
            deferred = _deferred_operands(runner.plan, runner.tensors)
            memory = runner.memory
            if terms is not None:
                terms.update(
                    term=runner.tensors["term"],
                    loss_weights=runner.tensors["loss_weights"],
                )
            del runner
    if binding is not None:
        loss, logp, dx_acc, dw_acc, deferred = binding.forward(
            X,
            W,
            labels,
            infer_logp,
            loss_weights,
            row_index,
            hc,
            stream_mode,
            capture=terms,
            dw_acc=arena,
        )
    hc = arena = None
    if terms is not None:
        # after the binding's per-call buffers went back to the allocator: only the two term buffers remain of them
        loss = _zero_weight_exact_loss(
            problem, terms, row_index, zero_index, loss_weights, backend=backend
        )
        terms = None
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
        exact=exact,
        deferred_alive=zero_pass and not zero_first and fuse,
    )


def _keep_runner_terms(runner: "LmHeadDetLossRunner") -> None:
    """A prepared (``reference`` backend) runner of a zero-weight split's chunk loop that keeps its loss terms
    (:func:`_term_rows`): its ``row_finalize`` / ``loss_reduce`` host values re-formed with ``keep_terms``."""
    t = runner.tensors
    t["keep_terms"] = True
    for key in runner.forward_order:
        if key[0] in ("row_finalize", "loss_reduce"):
            runner.values[key] = stage_values(key[0], t, runner.plan, key[1])


def _zero_weight_exact_loss(
    problem: Problem,
    terms: Optional[dict],
    row_index: torch.Tensor,
    zero_index: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    backend: str,
) -> torch.Tensor:
    """The FP32 ``[1]`` loss of a zero-weight split, bitwise the loss of the same call without the split.

    The plain path reduces the per-row terms ``term_t = w_t * min(ratio_t, 2)`` of the valid rows in its chunks of the
    valid rows: ``loss_reduce`` per chunk (one CTA; thread ``i`` adds ``term[i], term[i + 256], ..`` into an FP64
    register from ``+0``, a warp butterfly and a cross-warp butterfly, then the chunk sum goes into the FP64
    accumulator -- stored by the first chunk, added by the others -- and the last chunk stores ``float(0 - acc)``).
    This forms that array: ``term_v [T_v]`` (``T_v = |G| + |Z|``, the valid rows in ascending order) holds each
    gradient row's term -- the chunk loop's own ``row_finalize`` value (``terms``: chunk 0's in ``term``, chunk
    ``c >= 1``'s at ``loss_weights[(c - 1) C + r]``, :func:`_term_rows`) -- at its position among the valid rows,
    ``k + |{z < g_k}|``, and each zero-weight row's weight (``+0`` / ``-0``) at ``j + |{g < z_j}|``: the term the plain
    path's ``row_finalize`` forms for that row (``w * clipped`` with ``clipped`` in ``[0, 2]`` -- a NaN or infinite
    ratio clips to 2 -- so ``+-0`` with the weight's sign).  Then the shipped ``loss_reduce`` runs over the plain path's
    chunks of ``term_v`` (``plan_chunks(T_v, C)``: the same ``rows_c`` and first / last flags) on this call's stream:
    every thread adds the same values in the same order, so every partial sum, the accumulator and the rounding are the
    plain path's.  ``terms`` ``None``: no gradient row.  The ``reference`` backend runs its own ``loss_reduce`` over the
    same chunks (the plain reference path's bits).  Temporaries: ``term_v`` (``4 T_v`` bytes), the gradient rows'
    positions (``8 |G|``), then the zero-weight rows' positions and weights (``12 |Z|``)."""
    device = zero_index.device
    n_g = 0 if row_index is None else int(row_index.numel())
    n_z = int(zero_index.numel())
    n_v = n_g + n_z
    chunk = int(problem.chunk)
    ar = _split_arange(device, max(n_g, n_z))
    term_v = torch.empty((n_v,), dtype=torch.float32, device=device)
    weights = loss_weights.detach().index_select(0, zero_index)
    if n_g:
        # the k-th gradient row's position among the valid rows: k + (zero-weight rows ahead of it)
        pos = torch.searchsorted(zero_index, row_index)
        pos.add_(ar[:n_g]).sub_(1)
        head = min(chunk, n_g)
        term_v.index_copy_(0, pos[:head], terms["term"][:head])
        if n_g > head:
            term_v.index_copy_(0, pos[head:], terms["loss_weights"][: n_g - head])
        del pos
        terms.clear()
        pos = torch.searchsorted(row_index, zero_index)
        pos.add_(ar[:n_z]).sub_(1)
        term_v.index_copy_(0, pos, weights)
        del pos
    else:
        term_v.copy_(weights)
    del weights
    acc = torch.empty((1,), dtype=torch.float64, device=device)
    out = torch.empty((1,), dtype=torch.float32, device=device)
    chunks = plan_chunks(n_v, chunk)
    scalars = dict(
        mode=int(problem.mode),
        loss_div=float(problem.loss_div) if problem.loss_div is not None else 1.0,
    )

    def chunk_values(i: int) -> dict[str, Any]:
        row0, rows_c = chunks[i]
        return dict(
            term=term_v[row0 : row0 + rows_c],
            loss_acc=acc,
            loss_out=out,
            rows_c=int(rows_c),
            first_chunk=int(i == 0),
            last_chunk=int(i == len(chunks) - 1),
            **scalars,
        )

    if backend == "reference":
        for i in range(len(chunks)):
            ReferenceEngine.loss_reduce(chunk_values(i))
        return out
    module_name, record = record_for(device)
    record_abi(record)
    geometry = Geometry.from_record(record)
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    with _ffi_stream_context(index):
        for i in range(len(chunks)):
            values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
            values.update({name: 0 for name in COMMON_SCALARS})
            values.update(chunk_values(i))
            launch, arguments = _eager_launch(
                record, module_name, "loss_reduce", values, device, geometry
            )
            if launch.prepare is not None:
                launch.prepare(*arguments)
            launch.entry(*arguments)
    return out


def _zero_weight_logp(
    problem: Problem,
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    zero_index: torch.Tensor,
    *,
    backend: str,
    gemm_tuning: dict[str, dict[str, Any]],
    arena: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """``(logp, memory)`` of the zero-weight rows ``zero_index`` of a loss call: the log-probability forward's chunk
    loop compacted to those rows (:func:`_forward_logprob_rows`: ``gemm_logits_lbl`` + ``row_finalize`` where
    registered, chunk-bounded buffers, the call's ``chunk_size`` and logits GEMM knobs, no count, no host
    synchronization).  ``logp`` is compact (``[|Z|]``); per row the same logits and row-statistics arithmetic as the
    loss path's, so the same bits.  ``arena`` (:func:`_arena_views`): the pass's chunk buffers are views of its storage."""
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
        arena=arena,
    )
    return logp, memory


# Temporaries of the chunk loop that coexist with the zero-weight rows' logp-only pass in either order: the compaction
# outputs formed before both (:func:`_with_zero_weight_rows`); with ``chunk0_alive`` also the hidden valid-row count's
# chunk-0 buffers (the pass ahead of the loop of a call that defers its last chunk's weight gradient).
_ZERO_PASS_HELD = ("row_index", "row_valid")
_ZERO_PASS_CHUNK0 = ("logits", "stats", "x_c")


def _arena_views(
    arena: Optional[torch.Tensor],
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> Optional[dict[str, torch.Tensor]]:
    """``name -> view`` of ``arena``'s storage for each ``(shape, dtype)`` of ``specs`` (offsets 256-byte aligned), or
    ``None`` when they do not fit (or without ``arena``).  The arena is the FP32 ``[V, H]`` weight-gradient accumulator
    of a zero-weight split's chunk loop, allocated ahead of the logp-only pass (:func:`forward_loss`): an empty tensor
    the loop writes before it reads it, so the pass's chunk buffers live in it and the pass leaves no freed allocator
    segment the loop's larger allocations cannot reuse."""
    if arena is None:
        return None
    raw = arena.reshape(-1).view(torch.uint8)
    views, offset = {}, 0
    for name, (shape, dtype) in specs.items():
        size = math.prod(shape) * dtype.itemsize
        if offset + size > raw.numel():
            return None
        views[name] = raw[offset : offset + size].view(dtype).view(shape)
        offset += -(-size // 256) * 256
    return views


def _carved_bytes(layout: dict) -> int:
    """The arena bytes :func:`_arena_views` takes for the chunk buffers of ``layout`` (:func:`workspace_layout`)."""
    return sum(
        -(-int(layout[name][1]) // 256) * 256
        for name in _ZERO_PASS_CHUNK0
        if name in layout
    )


def _with_zero_weight_rows(
    result: Any,
    zero_index: Optional[torch.Tensor],
    zero_logp: Optional[tuple[torch.Tensor, dict]],
    *,
    chunk0_alive: bool = False,
    exact: bool = False,
    deferred_alive: bool = False,
) -> Any:
    """A loss forward's result with its zero-weight rows: their ``logp`` (when the logp-only pass ran) scattered into
    the ``[T]`` output, and the memory report's ``zero_weight_rows`` / ``zero_weight_logp_temporary_bytes`` /
    ``zero_weight_loss_mode`` (only when the call had zero-weight rows) and, with the exact loss,
    ``zero_weight_exact_loss_temporary_bytes`` (:func:`_zero_weight_exact_loss`).  With the pass, the report's
    ``peak_temporary_bytes`` is the larger of the chunk loop's and the pass's phase -- its own temporaries plus the
    loop's that coexist with them (the compaction outputs; with ``chunk0_alive``, the pass ran ahead of the loop while
    the hidden valid-row count held chunk 0's logits / statistics / gather buffers, which count too; with
    ``deferred_alive``, the pass ran after a loop whose deferred last chunk keeps its ``dz`` rows, ``saved_dz_bytes``,
    which count too)."""
    if zero_index is None:
        return result
    memory = dict(result.memory or {})
    n_z = int(zero_index.numel())
    n_g = 0 if result.row_index is None else int(result.row_index.numel())
    memory["zero_weight_rows"] = n_z
    memory["zero_weight_loss_mode"] = "exact" if exact else "regroup"
    if exact:
        memory["zero_weight_exact_loss_temporary_bytes"] = 4 * (n_g + n_z) + max(
            8 * n_g, 12 * n_z
        )
    zero_temp = 0 if zero_logp is None else int(zero_logp[1].get("temporary_bytes", 0))
    memory["zero_weight_logp_temporary_bytes"] = zero_temp
    if zero_logp is not None:
        temporary = memory.get("temporary") or {}
        held = sum(int(temporary.get(name, 0)) for name in _ZERO_PASS_HELD)
        if chunk0_alive:
            held += sum(int(temporary.get(name, 0)) for name in _ZERO_PASS_CHUNK0)
        if deferred_alive:
            held += int(memory.get("saved_dz_bytes", 0))
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
    :func:`zero_dlogp_partition`): the inclusive int32 scan of the live rows ``acc`` holds first, already formed
    (``row_valid`` is then not read, no ``cumsum``): one :func:`scale_cast_scatter` launch reads the accumulator's
    live rows only -- its rows past the live count (formed from stale ``dz`` rows by the uncounted dX GEMM) are never
    read -- and writes exact zeros on every other output row."""
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
    deferred weight-gradient GEMM's form (:func:`dw_cast`).  Pass only the accumulators of this package's
    :func:`forward_loss`: raw FP32 accumulators carry no mode tag, so this function cannot refuse those of another
    package (the non-deterministic package's included)."""
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
    keep: Optional[Callable[[Optional[torch.Tensor]], bool]] = None,
) -> ForwardResult:
    """Forward of the log-probability entry: ``logp`` (FP32 ``[T]``, 0 on ignored rows) plus the saved statistic
    ``lse`` (compacted: the ``[T_v]`` rows of ``row_index``, exactly what :func:`backward_logprob` expects back,
    together with ``row_index``).  ``compact_rows`` (default :func:`compact_rows_default`), ``hidden_count`` (default
    :func:`hidden_count_default`) and ``gemm_tuning`` (default :func:`gemm_tuning_default`) as in
    :func:`forward_loss`; ``logprob_lbl`` (default :func:`logprob_lbl_default`) writes only the row statistics and
    the label column's logit from the logits GEMM (bitwise the same ``lse`` / ``logp``).  This forward has no
    weight gradient, so it takes no ``fuse_dw_cast`` / ``dx_finalize`` / ``dw_stream``.  ``keep`` (the opt-in module's
    one-chunk kept logits: its rule's predicate on the valid-row index; ``cake`` backend): a call that counts its valid
    rows on the host and that ``keep`` accepts on this count's index runs :func:`forward_logprob_keep`'s binding
    instead (a :class:`KeptForwardResult`, bitwise the same ``logp`` / ``lse``); the decision follows the count with no
    other work between them, and either form runs on that one count."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if keep is not None and backend != "cake":
        raise ValueError("the kept logits (logprob_keep) run on the cake backend")
    problem = validate_lm_head_det_inputs(
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
    if keep is not None and hc is None and keep(row_index):
        return _forward_keep_bound(
            problem,
            X,
            W,
            labels,
            row_index,
            None,
            chunk_size=chunk_size,
            backend=backend,
            tuning=tuning,
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
    return ForwardResult(
        loss=None,
        logp=scatter_rows(logp, row_index, T),
        lse=lse,
        memory=memory,
        backend=backend,
        row_index=row_index,
        num_rows=T,
    )


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
    arena: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """``(logp, lse, memory)`` of the log-probability forward's chunk loop over the rows ``row_index`` (``None`` =
    every row; non-empty otherwise), through the remembered binding or a new one: :func:`forward_logprob`'s body after
    its count, and the zero-weight rows' logp-only pass of a loss call (:func:`_zero_weight_logp`).  ``logp`` / ``lse``
    are compact when ``row_index`` is given.  ``arena`` (the zero-weight pass, :func:`_arena_views`): the chunk buffers are
    views of its storage, so the first call launches through the binding too (the runner's workspace is never
    written)."""
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
            logp, lse = binding.forward(
                X, W, labels, None, None, row_index, hc, arena=arena
            )
            return logp, lse, binding.memory
    runner = prepare_lm_head_det_loss(
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
        _chunk_views=hc,
    )
    if hc is not None or (arena is not None and backend == "cake"):
        # the hidden valid-row count: chunk 0's logits already sit in the prologue's buffers, so the first call launches
        # through the templated binding too -- a template whose chunk buffers are views of the prologue's
        # (``_chunk_views``: no second chunk buffer beside them); the same with an arena (the runner's own workspace is
        # never written)
        binding = _Binding.from_runner(runner, hc)
        del runner
        if key is not None:
            cache.remember(key, binding)
        logp, lse = binding.forward(
            X, W, labels, None, None, row_index, hc, arena=arena
        )
        return logp, lse, binding.memory
    logp, lse = runner.forward()
    if key is not None:
        cache.remember(key, _Binding.from_runner(runner))
    return logp, lse, runner.memory


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
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward of the log-probability entry: recompute each chunk's logits, ``dz = dlogp_t *
    (1[v = y_t] - softmax)`` (ignored rows zero), then ``dX`` (BF16) and ``dW`` (BF16) through
    the FP32 accumulators.  ``(None, None)`` when neither input is trainable.  ``compact_rows``
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

    Zero-gradient row skip (:func:`zero_dlogp_partition`; a compacted backward that reuses the forward's row index,
    with the ``dz`` arm and, when it produces ``dX``, the fused dX finalize, of at least :func:`zero_dlogp_min_rows`
    valid rows, on a record with the ``_dcnt`` forms): the chunk loop takes the valid rows whose ``dlogp`` is nonzero
    first (a stable partition: each group in ascending row order) and its dz and dW GEMMs bound their work by the
    per-chunk device count of those rows (no host synchronization; capturable); the dX GEMM computes every row of its
    chunk as without the skip.  The rows of a zero ``dlogp`` get exact-zero ``dX`` rows and add nothing to ``dW``; a
    live row's ``dX`` row is the same per-row arithmetic (its chunk may change, and with it the chunk's dX slice
    count); ``dW`` sums the same nonzero products over the live rows in the new row order (FP32).  Every output is the
    same on every run.  The call's plan decides: a plan whose chunks take a form without a counted one (an explicitly
    pinned weight-gradient TMA epilogue) runs the backward without the skip, rows in the forward's order."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_det_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        entry="logprob",
    )
    stream_mode = dw_stream_mode(dw_stream)
    tuning = _resolve_gemm_tuning(gemm_tuning)
    dz_arm = _resolve_logprob_dz(logprob_dz)
    if not (need_dx or need_dw):
        return None, None
    T, H, V = problem.num_rows, problem.hidden, problem.vocab
    fused_dx = bool(
        need_dx and _resolve_dx_finalize(dx_finalize)
    )  # resolved once per call
    compact = bool(T and _resolve_compact(compact_rows))
    hc = None
    # the zero-gradient row skip: eligible for a call that reuses the forward's row index, run when the call's plan has
    # its counted forms (``order_rows``)
    skip = False
    scan = counts = None
    if row_index is not _RECOUNT:
        # the forward's valid-row index: the plan is known without a count (the valid-row mask of the fused dX finalize
        # is one device comparison, no synchronization; an eligible call forms none: its finalize reads the
        # partition's scan, and a plan without the counted forms forms it in ``order_rows``)
        if row_index is not None and not compact:
            raise ValueError("a valid-row index belongs to a compacted backward")
        skip = _zero_dlogp_skip(X.device, row_index, backend, dz_arm, need_dx, fused_dx)
        row_valid = (
            (labels >= 0) if (row_index is not None and fused_dx and not skip) else None
        )
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
        return finalize_dx(
            dx, None, row_index, row_valid, T, backend=backend, scan=scan
        )

    def order_rows(counted: bool) -> None:
        # the plan decides the skip: its counted forms order the rows (one partition launch); a plan without them runs
        # the backward without the skip over the forward's row order (no partition, the fused finalize's valid-row
        # mask, formed here as the call without the skip forms it before its first launch)
        nonlocal skip, row_index, lse, scan, counts, row_valid
        skip = skip and counted
        if skip:
            row_index, lse, scan, counts = zero_dlogp_partition(
                labels, dlogp, row_index, lse, problem.chunk
            )
        elif row_valid is None and row_index is not None and fused_dx:
            row_valid = labels >= 0

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
            zero_dlogp=skip,
        )
        binding = cache.lookup(key)
        if binding is not None:
            # the key of an eligible call remembers the plan it took (counted, or the one without the counts)
            order_rows(binding.plan.dcnt)
            dx, dw = binding.backward_logprob(
                X, W, labels, lse, dlogp, row_index, hc, stream_mode, counts
            )
            # the reordered statistic is read by the chunk loop only: released before the dX output boundary
            lse = None
            return finish_dx(dx), dw
    runner = prepare_lm_head_det_loss(
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
        # an eligible call plans with the counted forms; the counts follow the plan (the runner binds this slot, the
        # call launches from its binding with the counts of the rows it ordered)
        dcnt=torch.empty(
            (-(-rows // problem.chunk),), dtype=torch.int32, device=X.device
        )
        if skip
        else None,
    )
    if hc is not None or (skip and runner.plan.dcnt):
        binding = _Binding.from_runner(runner, hc)
        # the runner's workspace and accumulators are released before the partition allocates its outputs and the
        # binding its per-call buffers (the skip launches from the binding with the rows it ordered)
        del runner
        if key is not None:
            cache.remember(key, binding)
        order_rows(binding.plan.dcnt)
        dx, dw = binding.backward_logprob(
            X, W, labels, lse, dlogp, row_index, hc, stream_mode, counts
        )
        lse = None
        return finish_dx(dx), dw
    order_rows(False)
    dx, dw = runner.backward()
    if key is not None:
        cache.remember(key, _Binding.from_runner(runner))
    # the runner's chunk workspace is not needed by the dX finalize: dropped before its output is allocated
    del runner
    return finish_dx(dx), dw


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
    problem = validate_lm_head_det_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        entry="logprob",
    )
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
            return result
    runner = prepare_lm_head_det_loss_temperature(
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
        _chunk_views=hc,
    )
    if hc is not None:
        # the hidden valid-row count: chunk 0's logits already sit in the prologue's buffers, so the first call launches
        # through the templated binding too -- a template whose chunk buffers are views of the prologue's
        # (``_chunk_views``: no second chunk buffer beside them)
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
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """:func:`backward_logprob` with a temperature (:func:`_backward_logprob` is the path without one).  Backward
    of the log-probability entry: recompute each chunk's logits, ``dz = dlogp_t *
    (1[v = y_t] - softmax)`` (ignored rows zero), then ``dX`` (BF16) and ``dW`` (BF16) through
    the FP32 accumulators.  ``(None, None)`` when neither input is trainable.  ``compact_rows``
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
    = 0``), the weight-gradient GEMMs their counted forms (:func:`make_plan_temperature`).  ``logp`` and a live row's
    ``dX`` arithmetic are the same as without the skip; ``dW`` sums the live rows in the new row order (FP32), so its
    bits may differ when a row is zero.  The skip needs the temperature itself: the call's ``temperature``, or with
    ``temperature_rows`` the forward's resolved temperature (the autograd Function passes both);
    ``temperature_rows`` alone runs the backward without the skip.  A call without a temperature (none, or a scalar
    whose FP32 value is 1.0) is :func:`backward_logprob`, the skip included."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_det_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        entry="logprob",
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
        )
    problem = _with_temperature(problem)
    stream_mode = dw_stream_mode(dw_stream)
    tuning = _resolve_gemm_tuning(gemm_tuning)
    dz_arm = _resolve_logprob_dz(logprob_dz)
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
    if row_index is not _RECOUNT:
        # the forward's valid-row index: the plan is known without a count (the valid-row mask of the fused dX finalize
        # is one device comparison, no synchronization; an eligible call forms none: its finalize reads the
        # partition's scan, and a plan without the counted forms forms it in ``order_rows``)
        if row_index is not None and not compact:
            raise ValueError("a valid-row index belongs to a compacted backward")
        skip = tau is not None and _zero_dlogp_skip(
            X.device, row_index, backend, dz_arm, need_dx, fused_dx
        )
        row_valid = (
            (labels >= 0) if (row_index is not None and fused_dx and not skip) else None
        )
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

    def tau_rows_of(index: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        # the forward's rows serve the forward's order and, for a scalar, any order; else the gather by ``index``
        if temperature_rows is not None and (
            index is forward_index or (tau is not None and tau.kind == "scalar")
        ):
            return temperature_rows
        return _temperature_rows(tau, index, rows, X.device)

    # the compact temperature of the chunk loop's rows, in the order of the call's row index (``order_rows``)
    tau_rows = tau_rows_of(row_index)
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
        return finalize_dx(
            dx, None, row_index, row_valid, T, backend=backend, scan=scan
        )

    def order_rows(counted: bool) -> None:
        # :func:`backward_logprob`'s: the plan decides the skip (its counted forms order the rows, one partition
        # launch; a plan without them runs over the forward's row order); the temperature rows follow the order
        nonlocal skip, row_index, lse, scan, counts, row_valid, tau_rows
        skip = skip and counted
        if skip:
            row_index, lse, scan, counts = zero_dlogp_partition(
                labels, dlogp, row_index, lse, problem.chunk
            )
            tau_rows = tau_rows_of(row_index)
        elif row_valid is None and row_index is not None and fused_dx:
            row_valid = labels >= 0

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
            zero_dlogp=skip,
        )
        # an eligible call's key also holds which temperature form of the dz arm its plan may take
        key = temperature_binding_key(key) + (
            (zero_dlogp_temperature_dz_counted(),) if skip else ()
        )
        binding = cache.lookup(key)
        if binding is not None:
            # the key of an eligible call remembers the plan it took (counted, or the one without the counts)
            order_rows(binding.plan.dcnt)
            dx, dw = cast(_TemperatureBinding, binding).backward_logprob(
                X,
                W,
                labels,
                lse,
                dlogp,
                row_index,
                hc,
                stream_mode,
                counts,
                temperature=tau_rows,
            )
            # the reordered statistic and temperature rows are read by the chunk loop only: released before the dX
            # output boundary
            lse = tau_rows = None
            return finish_dx(dx), dw
    runner = prepare_lm_head_det_loss_temperature(
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
        temperature=tau_rows,
        # an eligible call plans with the counted forms; the counts and the reordered rows follow the plan (the call
        # launches from its binding with the rows it ordered)
        dcnt=torch.empty(
            (-(-rows // problem.chunk),), dtype=torch.int32, device=X.device
        )
        if skip
        else None,
    )
    if hc is not None or (skip and runner.plan.dcnt):
        binding = cast(_TemperatureBinding, _TemperatureBinding.from_runner(runner, hc))
        # the runner's workspace and accumulators are released before the partition allocates its outputs and the
        # binding its per-call buffers
        del runner
        if key is not None:
            cache.remember(key, binding)
        order_rows(binding.plan.dcnt)
        dx, dw = binding.backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            row_index,
            hc,
            stream_mode,
            counts,
            temperature=tau_rows,
        )
        lse = tau_rows = None
        return finish_dx(dx), dw
    order_rows(False)
    dx, dw = runner.backward()
    if key is not None:
        cache.remember(key, _TemperatureBinding.from_runner(runner))
    # the runner's chunk workspace is not needed by the dX finalize: dropped before its output is allocated
    del runner
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
    row_index: Any = _RECOUNT,
) -> KeptForwardResult:
    """:func:`forward_logprob` that keeps every chunk's BF16 logits for the backward (``cake`` backend): the logits
    GEMM is the plain ``gemm_logits`` (the whole ``z_c``, not the label-column arm), chunk 0's logits stay in the
    chunk buffer and the later chunks' go to one ``[T_v - rows_0, V]`` buffer, ``2 * T_v * V`` bytes in all
    (:attr:`KeptForwardResult.logits`).  Pass them to :func:`backward_logprob_keep`, which forms ``dz`` from them
    instead of recomputing the logits.  Bitwise the same ``logp`` / ``lse`` as :func:`forward_logprob`.
    ``row_index``: the caller's valid-row index of a compacted call (the ascending int64 rows whose label is ``>= 0``,
    ``None`` when every row is valid; the opt-in module forms it on the device from ``num_valid_rows``) -- the call
    then counts nothing (no host synchronization, no hidden count); the default counts the valid rows itself."""
    if backend != "cake":
        raise ValueError("the kept logits (logprob_keep) run on the cake backend")
    problem = validate_lm_head_det_inputs(
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
    hc = None
    if row_index is _RECOUNT:
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
    elif row_index is not None:
        if not compact:
            raise ValueError("a valid-row index belongs to a compacted forward")
        if isinstance(row_index, torch.Tensor):
            row_index = row_index.contiguous()  # the dX finalize reads a dense index
    if row_index is not None and row_index.numel() == 0:  # every row ignored
        return _kept_result(
            _empty_forward(
                problem, X, need_dx=False, need_dw=False, row_index=row_index
            ),
            None,
        )
    return _forward_keep_bound(
        problem,
        X,
        W,
        labels,
        row_index,
        hc,
        chunk_size=chunk_size,
        backend=backend,
        tuning=tuning,
    )


def _forward_keep_bound(
    problem: Problem,
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    row_index: Optional[torch.Tensor],
    hc: Optional[_HiddenCount],
    *,
    chunk_size: int,
    backend: str,
    tuning: dict[str, dict[str, Any]],
) -> KeptForwardResult:
    """:func:`forward_logprob_keep`'s body after its count (``row_index`` non-empty, or ``None``): the kept binding from
    the cache or a new one, launched; also the kept form of :func:`forward_logprob` under its ``keep``."""
    T = problem.num_rows
    cache = BINDING_CACHE
    key = None
    binding: Optional[_KeptBinding] = None
    if cache.enabled:
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
            valid_rows=None if row_index is None else int(row_index.numel()),
            gemm_tuning=tuning,
            hidden_count=hc is not None,
            logprob_lbl=False,
        ) + ("keep",)
        found = cache.lookup(key)  # a kept key binds a kept binding only
        if isinstance(found, _KeptBinding):
            binding = found
    if binding is None:
        runner = prepare_lm_head_det_loss(
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
        )
        # the kept views are bound per call: every call launches through the templated binding
        binding = cast(_KeptBinding, _KeptBinding.from_runner(runner, hc))
        del runner
        if key is not None:
            cache.remember(key, binding)
    logp, lse, kept = binding.forward_keep(X, W, labels, row_index, hc)
    return KeptForwardResult(
        loss=None,
        logp=scatter_rows(logp, row_index, T),
        lse=lse,
        memory=binding.memory,
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
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """:func:`backward_logprob` over the kept logits ``logits`` of :func:`forward_logprob_keep`
    (:attr:`KeptForwardResult.logits`, ``cake`` backend): each chunk's ``dz`` is formed by ``row_grad`` over its kept
    view in place -- no logits GEMM; the chunk's ``X`` rows are gathered for the weight gradient only -- then ``dX`` /
    ``dW`` as there (bitwise the same).  The views serve one backward: this call marks them consumed and a second
    backward on them is a ``ValueError``.  ``row_index`` as in :func:`backward_logprob`
    (the forward's valid-row index, or a recount); the hidden valid-row count and ``logprob_dz`` do not apply."""
    if backend != "cake":
        raise ValueError("the kept logits (logprob_keep) run on the cake backend")
    problem = validate_lm_head_det_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        entry="logprob",
    )
    stream_mode = dw_stream_mode(dw_stream)
    tuning = _resolve_gemm_tuning(gemm_tuning)
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
    dlogp = dlogp.detach().reshape(T).to(torch.float32).contiguous()
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    dx_cast = not (fused_dx and row_index is not None)
    cache = BINDING_CACHE
    key = None
    binding: Optional[_KeptBinding] = None
    if cache.enabled:
        key = logprob_backward_binding_key(
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
        ) + ("keep",)
        found = cache.lookup(key)  # a kept key binds a kept binding only
        if isinstance(found, _KeptBinding):
            binding = found
    if binding is None:
        runner = prepare_lm_head_det_loss(
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
            logprob_keep=True,
        )
        binding = cast(_KeptBinding, _KeptBinding.from_runner(runner))
        del runner
        if key is not None:
            cache.remember(key, binding)
    views = check_kept_logits(binding.plan, logits)
    for view in views:  # consumed from here on: row_grad overwrites them with dz
        setattr(view, _KEPT_SPENT, True)
    dx, dw = binding.backward_logprob_keep(
        X, W, labels, lse, dlogp, views, row_index, stream_mode
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
# accumulators (one gradient pass per step instead of two).  CPU offload or other saved-tensor hooks innermost and a
# torch whose checkpoint lacks the hook structure read below keep the fused forward.  The log-probability entry forms
# no gradient in its forward; under ``"auto"`` its checkpoint recomputation takes the checkpointed call's row results
# (:class:`_LogprobKept`).  ``"forward"``: every call with an input that requires a gradient forms
# them in its forward, also when no backward follows.  The outputs are the same bits either way (the same kernels on
# the same operands, in the same order).
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
# pass takes -- so while a selective checkpoint's mode is active every det autograd Function's forward runs with ALL
# active torch dispatch modes popped (:func:`_modes_off`): the selective checkpoint's, and any other on the stack (a
# user mode such as ``FlopCounterMode`` then does not see the entry's operators; a tracing mode such as
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
    """Whether a selective activation checkpoint's dispatch mode is active (read by every det autograd Function's
    forward under saved-tensor hooks; the forward then runs again with every active dispatch mode popped).  False on
    a torch without the private APIs read here (the entries then behave as before)."""
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
    leaves its gradients to the recomputation, and (a recomputed call) the checkpointed call it hands its result to.  A
    checkpointed call that does gets ``ctx.split`` (with the options its backward's own fused forward would take)."""
    key = _loss_call_key(
        X, W, labels, infer_logp, loss_weights, need_dx, need_dw, options
    )
    if ckpt.first:
        split = gradient_placement() == "auto"
        record = _checkpoint_first_call(ckpt, key, split)
        if split:
            calls, index = record if record is not None else (None, None)
            ctx.split = _SplitCall(calls, index, None)
            ctx.options, ctx.need = options, (need_dx, need_dw)
        return split, None
    # the recomputation follows its checkpointed call (unknown: the placement decides, as it did there)
    record = _checkpoint_recomputed_call(ckpt)
    if record is None:
        return gradient_placement() == "auto", None
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
    inputs.  ``None``: the fused forward (other hooks on top, an unsupported torch, the placement ``"forward"``)."""
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
    result = forward_loss(
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
    state = _split_state(ctx, ctx.saved_tensors, forward_loss)
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


class CakeLMHeadDetLossFunction(torch.autograd.Function):
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
        # under saved-tensor hooks: a non-reentrant torch.utils.checkpoint's passes leave the gradients to the
        # recomputation (gradient placement "auto"); the plain step (no hooks) and other hooks run the fused forward.
        # Under a selective checkpoint's modes (any need_dx / need_dw): again, with the dispatch modes popped
        if _TOP_SAVED_HOOKS(False) is not None and (
            need_dx or need_dw or _selective_checkpoint_modes()
        ):
            if _selective_checkpoint_modes():
                with _modes_off():
                    return CakeLMHeadDetLossFunction.forward(
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
                ),
            )
            if out is not None:
                return out
        result = forward_loss(
            X,
            W,
            labels,
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
        ctx.empty = X.shape[0] == 0 or (
            result.row_index is not None and result.row_index.numel() == 0
        )
        ctx.split = None
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


class CakeLMHeadDetLogprobFunction(torch.autograd.Function):
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
                return CakeLMHeadDetLogprobFunction.forward(
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
            if ckpt is not None and ckpt.first:
                _logprob_checkpoint_keep(ctx, ckpt, key, inputs, logp, (lse, row_index))
        # the valid-row index rides along: the backward plans its chunks without counting the rows again
        ctx.save_for_backward(X, W, labels, lse, row_index)
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
        dx, dw = _logprob_backward(ctx, dlogp, ctx.saved_tensors)
        return (dx, dw) + (None,) * 11


def _logprob_backward(ctx, dlogp, saved) -> tuple:
    """``(dX, dW)`` of :class:`CakeLMHeadDetLogprobFunction`'s backward from its saved tensors ``saved`` (``X``,
    ``W``, ``labels``, ``lse``, ``row_index``), unpacked once by the caller (a subclass that reads them first, the
    package's opt-in automatic form among them; a second unpack inside a non-reentrant checkpoint raises)."""
    X, W, labels, lse, row_index = saved
    need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
    return backward_logprob(
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
    )


class CakeLMHeadDetLogprobTemperatureFunction(torch.autograd.Function):
    """:class:`CakeLMHeadDetLogprobFunction` of a call with a temperature (:func:`cake_lm_head_det_train_logprob`)."""

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
                return CakeLMHeadDetLogprobTemperatureFunction.forward(
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
        dx, dw = _logprob_temperature_backward(ctx, dlogp, ctx.saved_tensors)
        return (dx, dw) + (None,) * 13


def _logprob_temperature_backward(ctx, dlogp, saved) -> tuple:
    """``(dX, dW)`` of :class:`CakeLMHeadDetLogprobTemperatureFunction`'s backward from its saved tensors ``saved``
    (``X``, ``W``, ``labels``, ``lse``, ``row_index``, ``tau_rows``, ``tau_tokens``), unpacked once by the caller (as
    :func:`_logprob_backward`)."""
    X, W, labels, lse, row_index, tau_rows, tau_tokens = saved
    # the forward's resolved temperature (None: the forward ran without one)
    tau = (
        _Temperature("tensor", tau_tokens)
        if tau_tokens is not None
        else _Temperature("scalar", ctx.temperature_scalar)
        if ctx.temperature_scalar is not None
        else None
    )
    need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
    return backward_logprob_temperature(
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
    )


class CakeLMHeadDetLogprobKeepFunction(torch.autograd.Function):
    """Entry (b) with the kept logits (:func:`cake_lm_head_det_train_logprob_keep`): the forward saves the row statistic
    and the kept logits; the first backward forms ``dz`` from them, a later one (a retained graph) recomputes.
    ``row_index`` (optional, last): the caller's valid-row index (:func:`forward_logprob_keep`; the opt-in module's
    ``num_valid_rows``); omitted, the forward counts the valid rows itself.  The backward returns one gradient slot per
    input either way (PyTorch drops trailing ``None`` slots past the inputs given)."""

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
        row_index=_RECOUNT,
    ):
        if _TOP_SAVED_HOOKS(False) is not None and _selective_checkpoint_modes():
            with _modes_off():  # under a selective checkpoint's modes: popped (see _selective_checkpoint_modes)
                return CakeLMHeadDetLogprobKeepFunction.forward(
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
                    row_index,
                )
        compact = _resolve_compact(compact_rows)
        hidden = _resolve_hidden_count(hidden_count)
        tuning = _resolve_gemm_tuning(gemm_tuning)
        result = forward_logprob_keep(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            compact_rows=compact,
            hidden_count=hidden,
            gemm_tuning=tuning,
            row_index=row_index,
        )
        ctx.set_materialize_grads(False)
        # the kept logits are saved tensors too, so a non-reentrant checkpoint drops them with the rest; ``row_grad``
        # overwrites them with ``dz`` through a raw pointer (no version-counter bump), so the first backward spends
        # them (``ctx.kept`` True -> False) and a later one through a retained graph recomputes (bitwise the same)
        ctx.save_for_backward(
            X, W, labels, result.lse, result.row_index, *(result.logits or ())
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
            return (None,) * 12
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
            dx, dw = backward_logprob_keep(
                X, W, labels, lse, dlogp, tuple(kept), **common
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
        return (dx, dw) + (None,) * 10


def cake_lm_head_det_train_loss(
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
    * ``dx_finalize`` (default ``True``, :func:`dx_finalize_default`): a sliced
      dX GEMM's slabs are added with one kernel and a compacted call's
      ``[T, H]`` ``dX`` is written in one pass (bitwise the same ``dX``).
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

    Gradient placement (:func:`set_gradient_placement`, default ``"auto"``): a
    call that no backward can follow (grad mode off, also the first pass of a
    reentrant ``torch.utils.checkpoint``, or no input requiring a gradient)
    computes ``loss`` (and ``logp``) only, without a gradient GEMM; inside a
    non-reentrant ``torch.utils.checkpoint`` the checkpointed forward computes
    the loss only and the recomputation in the backward forms the gradient
    accumulators, so a checkpointed step runs the gradient GEMMs once (CPU
    offload or other saved-tensor hooks innermost keep the fused forward).
    Under selective activation checkpointing the call runs with every active
    torch dispatch mode popped (``_selective_checkpoint_modes``) and behaves
    the same way.
    The outputs are the same bits as with ``"forward"`` (gradients formed in
    every forward).

    Zero-weight policy rows (a compacted policy call of at least
    :func:`zero_weight_min_rows` rows, :func:`forward_loss`): the rows whose
    ``loss_weights`` entry is ``+0`` / ``-0`` stay out of the gradient GEMMs
    (their ``dX`` rows are exact ``+0``); with ``return_logp`` their ``logp``
    comes from a logp-only pass (the same bits); the loss follows
    :func:`set_zero_weight_loss_mode` (default ``"exact"``: the bits of the
    call without the split).
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
        result = forward_loss(
            X,
            W,
            labels,
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
    loss, logp = CakeLMHeadDetLossFunction.apply(
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


def cake_lm_head_det_train_logprob(
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
    their defaults (``None`` = the documented default) as in :func:`cake_lm_head_det_train_loss`, plus the logits GEMM arms
    of this entry: ``logprob_lbl`` (default ``True``, :func:`logprob_lbl_default`) -- the forward's logits GEMM
    writes the row statistics and the label column's logit only -- and ``logprob_dz`` (default ``True``,
    :func:`logprob_dz_default`) -- the backward's recompute writes ``dz`` from its epilogue, no ``row_grad`` launch
    (both bitwise the same outputs).  Every switch is resolved once in the forward; the backward reuses the
    forward's valid-row index (no count, no host synchronization) and recomputes under the same values."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    return CakeLMHeadDetLogprobFunction.apply(
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


def cake_lm_head_det_train_logprob_temperature(
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
    their defaults (``None`` = the documented default) as in :func:`cake_lm_head_det_train_loss`, plus the logits GEMM arms
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
    if temperature is None or (type(temperature) is float and temperature == 1.0):
        # the default path: no temperature operation (a scalar 1.0 resolves to none)
        return CakeLMHeadDetLogprobFunction.apply(
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
    return CakeLMHeadDetLogprobTemperatureFunction.apply(
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


def cake_lm_head_det_train_logprob_keep(
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
    """:func:`cake_lm_head_det_train_logprob` with the kept logits (opt-in; the opt-in module's
    ``memory_mode="keep_logits"``).  Where a backward can follow -- grad mode on and ``X`` or ``W`` requiring a
    gradient, ``cake`` backend -- the forward keeps every chunk's BF16 logits as saved tensors (``2 * T_v * V``
    bytes until the backward; :func:`forward_logprob_keep`) and the first backward forms ``dz`` from them with
    ``row_grad`` instead of recomputing the logits (three GEMMs per chunk instead of four); a later backward through
    a retained graph recomputes.  Elsewhere -- grad mode off (``torch.no_grad``, ``torch.inference_mode``, the first
    pass of a reentrant ``torch.utils.checkpoint``), no input requiring a gradient, the reference backend -- the call
    is :func:`cake_lm_head_det_train_logprob` (the label-column forward).  A forward with a trainable input whose
    backward never runs pays the full-logits forward and holds the kept logits until the graph is freed.  Bitwise
    the same ``logp`` / ``dX`` / ``dW`` as the default entry."""
    if (
        backend == "cake"
        and torch.is_grad_enabled()
        and (X.requires_grad or W.requires_grad)
    ):
        return CakeLMHeadDetLogprobKeepFunction.apply(
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
    return cake_lm_head_det_train_logprob(
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
