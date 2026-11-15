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

Opt-in forms of the Cake chunked LM-head + loss backend (:mod:`.cake_backend`):

* ``memory_mode="budget"``: the step peak (forward through backward, outputs
  included) stays within the bytes of a head that materializes the BF16
  logits of the valid rows (:func:`memory_plan`);
* ``num_valid_rows=<host integer>``: the forward plans on the caller's
  valid-row count instead of reading the count back from the device
  (:func:`given_valid_rows`);
* ``memory_mode="keep_logits"`` (log-probability entry): :mod:`.cake_backend`'s
  ``cake_lm_head_det_train_logprob_keep`` -- where a backward can follow, the
  forward keeps the BF16 logits of the valid rows (``2 * T_v * V`` bytes) and
  the backward forms ``dz`` from them instead of recomputing them; with
  ``num_valid_rows`` its forward runs on the index formed from the caller's
  count (no readback).

The default ``memory_mode`` (``None``, :data:`MEMORY_AUTO`) picks per call: a
call whose backward forms a BF16 weight gradient on the ``cake`` backend on
CUDA (grad mode on, ``W`` requiring a gradient) runs the fast form while its
predicted step peak stays within :func:`memory_plan`'s budget and the form
``"budget"`` picks when not and one fits (:func:`auto_memory_plan`) -- the loss
entry through this module's autograd function, the log-probability entries
without ``num_valid_rows`` through :mod:`.cake_backend`'s forward with a
backward that decides (:class:`CakeLMHeadDetLogprobAutoFunction`), with it
through this module's autograd functions.  A default log-probability call
without a temperature, of at most two chunks of rows, whose valid rows fit one
chunk keeps their BF16 logits
from its forward to its backward where the fast backward would run no
zero-gradient row skip (:func:`_one_chunk_keep_rule`, also with a frozen ``W``;
:class:`CakeLMHeadDetLogprobOneChunkFunction`; bitwise the fast form's
outputs).  A
call without ``num_valid_rows`` that forms no such gradient, and an explicit
``memory_mode="fast"`` without it, runs :mod:`.cake_backend`'s entry
unchanged.  The forms here share that module's kernels, launch helpers and
per-device state; the planner, the binding (with its own
:data:`BINDING_CACHE`), the forward, the backward and the autograd functions
are this module's.  Beyond the two keywords they differ from
:mod:`.cake_backend` in buffer lifetimes only:

* with ``memory_mode="budget"`` (and a default call whose automatic form is a
  budget form), a backward of the loss entry that frees the graph releases the
  forward's saved accumulators as soon as it has used them (a second backward
  through that graph raises; ``retain_graph=True`` keeps them);
* a call that misses the binding cache binds on buffers that share one
  allocation of the largest of them -- with the hidden valid-row count, on
  views of the count's chunk-0 buffers for the chunk buffers -- and never
  launches on them.

Every output is bitwise the same as :mod:`.cake_backend`'s, with two
exceptions of a default call whose automatic form is a budget form (both within
the accuracy budgets, the same bits on every run): a log-probability backward
runs without :mod:`.cake_backend`'s zero-gradient row skip, so its ``dX`` /
``dW`` bits can differ from the fast form's where valid rows have a zero
``dlogp``; a loss call with zero-weight policy rows decides its split as
``"budget"`` does (below), which can differ from the fast form's decision.  The
zero-weight
policy rows of :mod:`.cake_backend`'s loss entry are left out of the gradient
path here too -- in the fast form and with ``memory_mode="budget"``, whose
plan then runs over the gradient rows while its budget still counts every
valid row and its model adds the zero-weight rows' logp-only pass and the
exact loss's term phase -- but not with ``num_valid_rows``: a caller-given
count carries no split of the rows, so that form keeps every valid row in the
chunk loop.  The split keeps :mod:`.cake_backend`'s memory rule: its dX K
slices are capped at the plain call's and a call whose predicted split peak
exceeds the plain call's (the fast form's byte model, or the two budget plans)
runs without the split.  This module reads no environment variable.
"""

from __future__ import annotations

import functools
import math
from collections import OrderedDict
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, ClassVar, NamedTuple, Optional, Union, cast

import torch

from . import cake_backend as _main
from .cake_jit import (
    GEMM_STAGES,
    STAGES,
    load_cake_lm_head_det_loss_module,
    registered_stages,
    registered_stages_with_temperature,
    toolchain_runs_hidden_count,
)
from .cake_backend import (
    BACKENDS,
    BINDING_CACHE_DEFAULT_CAPACITY,
    COMMON_SCALARS,
    COMMON_TENSORS,
    ChunkVariants,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_GEOMETRY,
    DW_CAST_STAGES,
    ENTRIES,
    GRAD_WEIGHT_DTYPES,
    Geometry,
    HIDDEN_COUNT_MIN_CHUNKS,
    HIDDEN_COUNT_STAGE,
    IGNORE_INDEX,
    LOGITS_BASES,
    LOGPROB_DZ_STAGE,
    LOGPROB_DZ_TAU_STAGE,
    LOGPROB_LBL_STAGE,
    TEMPERATURE_STAGES,
    Problem,
    ReferenceEngine,
    TemperatureForm,
    _HiddenCount,
    _Launch,
    _OWNED_VALUES,
    _RECOUNT,
    _TemperatureGeometry,
    _assert_on_device,
    CHUNK_VIEW_REGIONS,
    _carve,
    _check_output,
    _chunk_view,
    _chunk_view_layout,
    _check_temperature_arms,
    _check_temperature_operands,
    _deferred_operands,
    _device_constants,
    _dw_stream,
    _dx_keys,
    _dx_reduce,
    _evict,
    _ffi_stream_context,
    _gather_rows,
    _hidden_count_arange,
    _hidden_count_cell,
    _hidden_count_memory,
    _hidden_count_pair_cell,
    _term_rows,
    _with_zero_weight_rows,
    _zero_weight_exact_loss,
    _valid_rows_of_count,
    _zero_weight_logp,
    _is_zero_accumulator,
    _zero_accumulator,
    _zero_gradient,
    gradient_index,
    gradient_rows,
    zero_weight_loss_mode,
    zero_weight_plan,
    zero_weight_split,
    _layout,
    _logprob_arm,
    _meta,
    _mm_fp32,
    _prepare_labels,
    _prepare_x,
    _reads_labels,
    _record_scratch_bytes,
    _record_tma_bytes,
    _resolve_compact,
    _resolve_dx_finalize,
    _resolve_fuse,
    _resolve_gemm_tuning,
    _resolve_hidden_count,
    _resolve_logprob_dz,
    _resolve_logprob_lbl,
    _resolve_temperature,
    _run_keys,
    _run_keys_temperature,
    _same_view,
    _temperature_rows,
    _tuning_key,
    _unit_scale,
    _values,
    _with_temperature,
    arch_for,
    base_stage,
    bind_stage,
    chunk_variants,
    cluster_resident,
    compaction_index,
    dw_block_variant,
    dw_cast_tma_rule,
    dw_stream_mode,
    dx_max_slices,
    dx_slices_fit,
    dx_wide_items,
    forward_early_keys,
    gemm_scalars,
    grid_dims,
    hidden_count_eligible,
    hidden_count_eligible_temperature,
    memory_report,
    plan_chunks,
    raster_variant,
    capped_k_slices,
    recommended_k_slices,
    recompute_keys,
    record_abi,
    record_for,
    scatter_rows,
    stage_form,
    stage_variant,
    temperature_binding_key,
    valid_row_index,
    valid_rows,
    validate_lm_head_det_inputs,
    workspace_layout,
    _hidden_count_memory_temperature,
    _logprob_arm_temperature,
    memory_report_temperature,
    _reads_labels_temperature,
    _recompute_keys_temperature,
)

MEMORY_MODES = (
    "fast",
    "budget",
)  # ``memory_mode`` of the autograd entries: the fastest form / the step-peak budget (:func:`memory_plan`)
# ``memory_mode`` of :func:`cake_lm_head_det_train_logprob` only: keep the forward's BF16 logits for the backward
# (``2 * T_v * V`` bytes), whose ``row_grad`` then forms ``dz`` from them instead of recomputing the logits --
# :mod:`.cake_backend`'s ``cake_lm_head_det_train_logprob_keep``
KEEP_LOGITS = "keep_logits"
# The default of the autograd entries (``memory_mode=None``, :func:`memory_mode_default`; the resolver accepts it, it
# is not one of :data:`MEMORY_MODES`): per call, the fast form while its predicted step peak stays within the budget of
# :func:`memory_plan`, else a budget form that fits, else the fast form (:func:`auto_memory_plan`)
MEMORY_AUTO = "auto"
# The automatic form plans a call only when its fast form runs at least this many chunks (over the rows it chunks
# over); a shorter call keeps the fast form without planning (:func:`_auto_plans`).  ``2``: a one-chunk call (no FP32
# ``[V, H]`` accumulator; its fast step peak exceeds the bound only by its outputs and gathered rows) keeps the fast form.
_AUTO_MIN_CHUNKS = 2


# The budget form's margin below the logits-materializing head's bytes (a fraction of them): the peak model of
# :func:`memory_plan` counts the vocabulary- and hidden-sized buffers, estimates the O(T) row vectors and adds the
# caching allocator's per-tensor bound (:func:`_allocated_bytes`); the margin absorbs the row-vector estimate and the
# small tensors the model does not list.
MEMORY_BUDGET_MARGIN = 0.02

# PyTorch's CUDA caching allocator under its default configuration (``PYTORCH_CUDA_ALLOC_CONF`` unset), which
# ``torch.cuda.max_memory_allocated`` counts in blocks: a block is a multiple of 512 bytes, and a large-pool block (a
# request of 1 MiB or more, counted conservatively) may keep an unsplit remainder of up to 1 MiB (``kSmallSize``).
ALLOCATOR_ROUND_BYTES = 512
ALLOCATOR_LARGE_BYTES = 1 << 20


def _allocated_bytes(nbytes: int, count: int = 1) -> int:
    """The bytes :func:`memory_plan` counts for ``count`` live tensors of ``nbytes`` each: their bytes plus, per tensor,
    the default caching allocator's bound -- :data:`ALLOCATOR_ROUND_BYTES` of rounding and, for a tensor of
    :data:`ALLOCATOR_LARGE_BYTES` or more, an unsplit remainder of up to :data:`ALLOCATOR_LARGE_BYTES`."""
    nbytes, count = int(nbytes), int(count)
    if nbytes <= 0 or count <= 0:
        return 0
    large = ALLOCATOR_LARGE_BYTES if nbytes >= ALLOCATOR_LARGE_BYTES else 0
    return count * (nbytes + ALLOCATOR_ROUND_BYTES + large)


def _exact_loss_bytes(
    rows: int, buffer_rows: int, *, zero_rows: int, zero_total: int
) -> int:
    """The bytes :func:`memory_plan` counts for the exact loss of a split call
    (:func:`.cake_backend._zero_weight_exact_loss`), next to the chunk loop's held state: the term array of the valid
    rows (``4 (rows + zero_total)``), the zero-weight rows' gathered weights (``4 zero_total``) and index
    (``8 zero_total``) and, when their logp-only pass ran ahead of the loop, their compact ``logp`` (``4 zero_rows``);
    then either the gradient rows' positions (``8 rows``) with the two kept term buffers (``4 rows + 4 buffer_rows``),
    or after those are freed the zero-weight rows' positions (``8 zero_total``) -- the larger of the two."""
    live = _allocated_bytes
    R, Cb, Z, Zt = int(rows), int(buffer_rows), int(zero_rows), int(zero_total)
    held = live(4 * (R + Zt)) + live(4 * Zt) + live(8 * Zt) + live(4 * Z)
    return held + max(live(4 * R) + live(4 * Cb) + live(8 * R), live(8 * Zt))


def memory_mode_default() -> str:
    """Default of ``memory_mode`` (:data:`MEMORY_AUTO`, ``"auto"``): per call, the fastest form (``"fast"``,
    ``fuse_dw_cast`` as given) while its predicted step peak -- with its own buffer lifetimes -- stays within the
    budget of :func:`memory_plan`, else the form ``"budget"`` picks when one fits, else the fast form
    (:func:`auto_memory_plan`; a call that cannot run a budget form keeps the fast form).  ``"budget"`` picks, per
    call, the fastest exact form whose predicted step peak stays within the budget of :func:`memory_plan` -- the bytes
    of a head that materializes the BF16 logits of the valid rows -- recomputing part or all of the weight gradient in
    vocabulary blocks in the backward where it must (bitwise the same ``dW``).  ``"fast"``: always the fastest form."""
    return MEMORY_AUTO


def _resolve_memory_mode(memory_mode) -> str:
    """``memory_mode`` resolved: ``None`` -> :func:`memory_mode_default`; one of :data:`MEMORY_MODES` or
    :data:`MEMORY_AUTO`, else ``ValueError``."""
    mode = memory_mode_default() if memory_mode is None else memory_mode
    if not isinstance(mode, str) or (mode not in MEMORY_MODES and mode != MEMORY_AUTO):
        raise ValueError(
            f"memory_mode must be None or one of {MEMORY_MODES}, got {memory_mode!r}"
        )
    return mode


# PyTorch's current-graph-task query (a private API), resolved once: ``None`` when this PyTorch lacks it
_GRAPH_TASK_KEEP_GRAPH = getattr(
    getattr(torch._C, "_autograd", None), "_get_current_graph_task_keep_graph", None
)


def _keep_graph() -> bool:
    """Whether the running backward keeps its graph (``retain_graph=True``, ``create_graph=True``), read from PyTorch's
    current graph task (a private API: ``torch._C._autograd._get_current_graph_task_keep_graph``, resolved at import as
    ``_GRAPH_TASK_KEEP_GRAPH``).  ``True`` when that API is missing or cannot answer: the autograd entries then keep
    their saved state until the graph is freed, as before -- they release it early only in a backward known to be the
    graph's last."""
    getter = _GRAPH_TASK_KEEP_GRAPH
    if getter is None:
        return True
    try:
        return bool(getter())
    except Exception:  # noqa: BLE001 -- outside a graph task (or an API change): keep
        return True


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
    and chunk 0's device-count logits GEMM of a compacted call without the valid-row count on the host;
    the count is read back asynchronously (:meth:`_HiddenCount.finish`).

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
    if temperature is not None and not stats:
        raise ValueError(
            "the hidden valid-row count with a temperature runs the device-count GEMM with the statistics"
        )
    stage = stage_form(
        "gemm_logits" if stats else "gemm_logits_nostats",
        count=True,
        temperature=temperature is not None,
    )
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


def _given_count(value, num_rows: int) -> Optional[int]:
    """The caller's valid-row count ``num_valid_rows`` as a host integer (``None`` = not given): a Python or NumPy
    integer, or an integer CPU tensor of one element, in ``[0, num_rows]``.  A CUDA tensor is rejected -- reading it
    would synchronize, which is what the keyword avoids -- as are ``bool`` and floating-point values."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("num_valid_rows must be a scalar")
        if value.device.type != "cpu":
            raise ValueError(
                "num_valid_rows must be a host integer (a Python / NumPy integer or a CPU tensor): reading a "
                f"{value.device.type} tensor would synchronize with the device"
            )
        if (
            value.dtype.is_floating_point
            or value.dtype.is_complex
            or value.dtype == torch.bool
        ):
            raise ValueError(
                f"num_valid_rows must be an integer, got dtype {value.dtype}"
            )
        value = int(value.item())
    elif isinstance(value, bool) or not isinstance(value, int):
        if (
            type(value).__module__ != "numpy"
            or not hasattr(value, "dtype")
            or value.dtype.kind not in "iu"
        ):
            raise ValueError(
                f"num_valid_rows must be an integer, got {type(value).__name__}"
            )
    n = int(value)
    if not 0 <= n <= int(num_rows):
        raise ValueError(
            f"num_valid_rows must lie in [0, T = {int(num_rows)}], got {n}"
        )
    return n


def _given_count_readback(device: torch.device) -> bool:
    """Whether a caller-given valid-row count is checked by reading the device count back (one synchronization)
    instead of a device-side assertion: host tensors (free) and every ``sm_103a`` call whose code CUDA 13.0 built --
    the calls :func:`cake_jit.toolchain_runs_hidden_count` keeps on the host-count schedule, whose host gap the
    given-count path would otherwise remove."""
    if device.type != "cuda":
        return True
    arch = arch_for(device)
    return arch is not None and not toolchain_runs_hidden_count(arch)


def _check_given_count(count: int, actual: int) -> None:
    """``ValueError`` when the caller's valid-row count ``count`` differs from the ``actual`` labels ``>= 0`` (a
    count read back: :func:`given_valid_rows`, :func:`_check_count_on_host`)."""
    if actual != count:
        raise ValueError(
            f"num_valid_rows={count} but the labels hold {actual} valid rows (labels >= 0)"
        )


def _check_count_on_host(labels: torch.Tensor, count: int) -> None:
    """``check_num_valid_rows=True`` on a path that forms no valid-row index from the caller's count ``count`` (an
    uncompacted call, a backward given the forward's index, a backward without gradients): one readback of the labels
    ``>= 0`` and :func:`_check_given_count`'s ``ValueError`` on a mismatch (a compacted call's count is checked by
    :func:`given_valid_rows`)."""
    _check_given_count(count, int((labels >= 0).sum()))


def given_valid_rows(
    labels: torch.Tensor, count: int, *, mask: bool = False, check: bool = False
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """:func:`valid_rows` for a caller-given valid-row count ``count`` (``num_valid_rows``): the plan is known on the
    host without a readback and the index is formed on the device -- :func:`compaction_index` (``labels >= 0`` ->
    inclusive int32 scan -> ``searchsorted``, three kernels), whose first ``count`` slots are ``valid.nonzero()`` when
    the count is right.  Same returns as :func:`valid_rows` (``(None, None)`` when ``count == T``).

    The count is checked on the device: one ``torch._assert_async(scan[T - 1] == count)`` queued right behind the
    compaction, ahead of every kernel that reads the index, so a wrong count stops the stream with a device-side
    assertion (which, like an out-of-range index in PyTorch, ends the CUDA context) before any kernel reads past the
    compact buffers.  ``check=True`` (debug) reads the count back instead -- one synchronization -- and raises
    ``ValueError`` on a mismatch; so do host tensors and the calls of :func:`_given_count_readback`."""
    T = int(labels.numel())
    if T == 0:
        return None, None
    count = int(count)
    device = labels.device
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
        else None
    )
    valid, pos, idx_full = compaction_index(labels, ar=ar)
    total = pos[T - 1]
    if check or _given_count_readback(device):
        # the one synchronization of the checked path
        _check_given_count(count, int(total))
    else:
        _assert_on_device(
            total == count,
            "num_valid_rows differs from the number of valid labels (labels >= 0)",
        )
    if count == T:
        return None, None
    return idx_full[:count], (valid if mask else None)


# Host-side steps of the chunk loop (torch operators over the bound tensors, no kernel of the program, no allocation).
def _save_row_grad(values: dict[str, Any]) -> None:
    """Copy a chunk's per-row gradient scale ``d`` into its rows of ``d_rows`` (the loss entry's budget forms: the
    backward recomputes ``dz`` of the vocabulary rows the forward did not accumulate from it)."""
    values["out"].copy_(values["src"])


HOST_STAGES: dict[str, Callable[[dict[str, Any]], None]] = {
    "dx_reduce": _dx_reduce,
    "gather_rows": _gather_rows,
    "save_row_grad": _save_row_grad,
}


@dataclass(frozen=True)
class Plan(_main.Plan):
    """:class:`.cake_backend.Plan` with the two fields of the budget forms; the planning helpers of
    :mod:`.cake_backend` take it as theirs."""

    # the budget forms of ``memory_mode="budget"`` (:func:`memory_plan`): the chunk loop's weight-gradient accumulate
    # covers the vocabulary rows ``[0, dw_rows)`` only and leaves its FP32 ``[dw_rows, H]`` accumulator to the caller
    # (no cast, no ``dw_out``; ``None`` = every row, as before), and the loss entry's forward keeps every chunk's
    # per-row gradient scale ``d`` in the ``[rows]`` vector ``d_rows`` for the backward's recompute of the other rows
    dw_rows: Optional[int] = None
    keep_row_grad: bool = False
    # the budget form ``"fused_ip"`` (:func:`memory_plan`; deferred last chunk, BF16 ``dW``): the chunk loop's FP32
    # accumulator -- the loss entry's forward, the log-probability backward's -- is two allocations, ``dw_acc`` of
    # the vocabulary rows ``[0, dw_split)`` and ``dw_acc_hi`` of ``[dw_split, V)``, each chunk adding into both (two
    # launches of its accumulate form, the second keyed by :class:`_UpperRows`), so the backward's fused cast can write
    # the BF16 ``dW`` into ``dw_acc``'s own storage (:func:`_dw_cast_split`); ``None`` = one accumulator, as before
    dw_split: Optional[int] = None
    # the budget form ``"partial_ip"``: the split accumulator's upper part covers the rows ``[dw_split, dw_split_end)``
    # only (none when equal; the backward recomputes the rows ``[dw_split_end, V)``, :func:`_dw_vocab_blocks`);
    # ``None`` = up to ``V``
    dw_split_end: Optional[int] = None

    def dw_acc_rows(self, index) -> tuple[int, int]:
        """The vocabulary rows ``[v0, v1)`` the weight-gradient accumulate of launch index ``index`` adds into its FP32
        accumulator: ``[0, dw_rows)`` (``[0, V)`` without ``dw_rows``), or with :attr:`dw_split` ``[0, dw_split)`` for
        a chunk index and ``[dw_split, dw_split_end)`` (``dw_split_end`` defaults to ``V``) for its
        :class:`_UpperRows`."""
        V = int(self.problem.vocab)
        if self.dw_split is None:
            return 0, (V if self.dw_rows is None else int(self.dw_rows))
        if isinstance(index, _UpperRows):
            return int(self.dw_split), (
                V if self.dw_split_end is None else int(self.dw_split_end)
            )
        return 0, int(self.dw_split)

    @property
    def dw_split_upper(self) -> bool:
        """Whether the split accumulator has an upper part (:attr:`dw_split` below its end)."""
        return self.dw_split is not None and self.dw_acc_rows(_UpperRows(0))[1] > int(
            self.dw_split
        )


class _UpperRows(int):
    """The launch index of a chunk's second weight-gradient accumulate under a split accumulator
    (:attr:`Plan.dw_split`): the key ``(stage, _UpperRows(i))`` adds chunk ``i``'s ``dz`` columns ``[dw_split, V)`` into
    ``dw_acc_hi`` beside the key ``(stage, i)`` of the rows ``[0, dw_split)``.  An ``int`` for indexing (the chunk's
    rows, variants and accumulate form are chunk ``i``'s), never equal to the plain index: the two keys are two
    launches."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return type(other) is _UpperRows and int(self) == int(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(("dw_split", int(self)))

    def __repr__(self) -> str:
        return f"_UpperRows({int(self)})"


class _TemperaturePlan(Plan, _main._TemperaturePlan):
    """:class:`.cake_backend._TemperaturePlan` of this module's :class:`Plan` (:func:`make_plan_temperature`)."""

    temperature: ClassVar[bool] = True
    plain: ClassVar[type] = Plan

    def without_temperature(self) -> Plan:
        return cast(Plan, super().without_temperature())


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
    dw_rows: Optional[int] = None,
    keep_row_grad: bool = False,
    dx_slices_cap: Optional[int] = None,
    dw_split: Optional[int] = None,
    dw_split_end: Optional[int] = None,
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
    (:attr:`Plan.logprob_dz`, :attr:`Plan.logprob_lbl`).  ``dw_rows`` (an unfused plan with a weight gradient; a
    multiple of :func:`dw_block_step` up to ``V``) accumulates the vocabulary rows ``[0, dw_rows)`` only and leaves the
    FP32 accumulator uncast for the caller; ``keep_row_grad`` (loss entry) keeps every chunk's per-row gradient scale
    (:attr:`Plan.dw_rows`, :attr:`Plan.keep_row_grad`; the budget forms of :func:`memory_plan`).  ``dx_slices_cap``
    (the zero-weight split): no chunk's rule-picked dX slice count exceeds it (:func:`.cake_backend.capped_k_slices`; an
    explicit ``k_slices`` knob is not capped).  ``dw_split`` (a multi-chunk plan of either entry with the deferred last
    chunk and a BF16 ``dW``; a multiple of :func:`dw_block_step` strictly inside ``(0, V)``, at least half of ``V``):
    the chunk loop's FP32 accumulator splits at that vocabulary row (:attr:`Plan.dw_split`, the budget form
    ``"fused_ip"``); ``dw_split_end`` (with ``dw_split``; a multiple of :func:`dw_block_step` in ``[dw_split, V]``,
    default ``V``) ends its upper part there (:attr:`Plan.dw_split_end`, the budget form ``"partial_ip"``; the loss
    entry with ``keep_row_grad``)."""
    if (logprob_dz or logprob_lbl) and problem.entry != "logprob":
        raise ValueError("the logits GEMM arms belong to the log-probability entry")
    if keep_row_grad and problem.entry != "loss":
        raise ValueError("keep_row_grad belongs to the loss entry")
    if dw_rows is not None:
        dw_rows = int(dw_rows)
        step = dw_block_step(geometry)
        if not need_dw or fuse_dw_cast:
            raise ValueError(
                "dw_rows belongs to an unfused plan with a weight gradient (fuse_dw_cast=False, need_dw=True)"
            )
        if not 0 < dw_rows <= int(problem.vocab) or (
            dw_rows % step and dw_rows != int(problem.vocab)
        ):
            raise ValueError(
                f"dw_rows={dw_rows}: a positive multiple of {step} up to V = {problem.vocab}"
            )
    rows = problem.num_rows if valid_rows is None else int(valid_rows)
    chunks = plan_chunks(rows, problem.chunk)
    if dw_split is not None:
        dw_split = int(dw_split)
        step = dw_block_step(geometry)
        if not (
            need_dw
            and fuse_dw_cast
            and dw_rows is None
            and problem.grad_weight_dtype == torch.bfloat16
            and len(chunks) > 1
        ):
            raise ValueError(
                "dw_split belongs to a multi-chunk plan with the deferred last chunk (fuse_dw_cast=True, "
                "need_dw=True, no dw_rows) and a BF16 weight gradient"
            )
        if dw_split % step or not 2 * dw_split >= int(problem.vocab) > dw_split:
            raise ValueError(
                f"dw_split={dw_split}: a multiple of {step} in [V / 2, V) for V = {problem.vocab}"
            )
        if dw_split_end is not None:
            dw_split_end = int(dw_split_end)
            if not dw_split <= dw_split_end <= int(problem.vocab) or (
                dw_split_end % step and dw_split_end != int(problem.vocab)
            ):
                raise ValueError(
                    f"dw_split_end={dw_split_end}: a multiple of {step} in [dw_split, V] for dw_split = {dw_split}, "
                    f"V = {problem.vocab}"
                )
    elif dw_split_end is not None:
        raise ValueError("dw_split_end belongs to a plan with dw_split")
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
    if (
        variants
        and need_dw
        and fuse_dw_cast
        and not (problem.entry == "loss" and problem.grad_weight_dtype == torch.float32)
        and "epi_store" not in tuning.get("dw", {})
        and dw_cast_tma_rule(len(chunks), chunks[-1][1], arch)
    ):
        # the deferred last chunk's bf16 cast takes the TMA epilogue; an explicit ``epi_store`` keeps the shipped cast
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
    return Plan(
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
        dw_rows=dw_rows,
        keep_row_grad=bool(keep_row_grad),
        dw_split=dw_split,
        dw_split_end=dw_split_end,
    )


def make_plan_temperature(problem: Problem, **kw: Any) -> Plan:
    """:func:`make_plan` of a log-probability call with a temperature (:attr:`Problem.temperature`) on a registered
    program (``arch``) running both logits GEMM arms (``logprob_dz`` and ``logprob_lbl``): the plan runs the
    temperature forms (:class:`_TemperaturePlan`; the budget form fields as in :func:`make_plan`)."""
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
    return _TemperaturePlan(**{f.name: getattr(plan, f.name) for f in fields(plan)})


def dw_block_step(geometry: Geometry = DEFAULT_GEOMETRY) -> int:
    """The vocabulary granule of the budget forms (:func:`memory_plan`): ``dw_rows`` and every recompute block are a
    multiple of it -- whole CTA-pair row tiles of the weight-gradient GEMM (its ``M`` is the vocabulary) and whole
    work items, statistics tiles and vocabulary multiples of the logits GEMM that recomputes a block (its ``N``)."""
    return math.lcm(
        int(geometry.row_tile)
        * max(int(geometry.cta_group), int(geometry.dw_cluster_ctas)),
        int(geometry.logits_item_cols),
        int(geometry.stats_tile),
        int(geometry.vocab_multiple),
    )


def dw_split_rows(vocab: int, geometry: Geometry = DEFAULT_GEOMETRY) -> int:
    """The vocabulary row at which the budget form ``"fused_ip"`` splits the chunk loop's FP32 weight-gradient
    accumulator (:attr:`Plan.dw_split`): ``V / 2`` rounded up to a multiple of :func:`dw_block_step`, so the lower
    accumulator's ``4 * a * H`` bytes hold the BF16 ``[V, H]`` ``dW`` the backward writes into them
    (:func:`_dw_cast_split`).  The form needs ``a < V``."""
    step = dw_block_step(geometry)
    return -(-int(vocab) // (2 * step)) * step


def _dw_split_pieces(rows: int, step: int) -> tuple[tuple[int, int], ...]:
    """The vocabulary row pieces ``[m0, m1)`` of the lower accumulator's in-place fused cast (:func:`_dw_cast_split`)
    over its ``rows`` rows: a first piece of ``rows / 16`` rows rounded up to ``step`` (written through a temporary),
    then pieces that double up to ``rows``, each with ``m1 <= 2 * m0``.  The BF16 rows ``[m0, m1)`` of ``dW`` take the
    storage bytes ``[2 m0 H, 2 m1 H)``, which hold FP32 rows below ``m0`` only -- rows the earlier pieces have read --
    while the piece reads its own FP32 rows from byte ``4 m0 H`` on."""
    p0 = min(int(rows), max(int(step), -(-int(rows) // (16 * int(step))) * int(step)))
    bounds = [0, p0]
    while bounds[-1] < rows:
        bounds.append(min(int(rows), 2 * bounds[-1]))
    return tuple(zip(bounds[:-1], bounds[1:], strict=True))


@dataclass(frozen=True)
class MemoryPlan:
    """The weight-gradient form :func:`memory_plan` picked for one call of a budget form (``memory_mode="budget"``, or
    a default call whose automatic form switched to it, :func:`auto_memory_plan`).

    ``form``: ``"fused"`` (the fast form: the deferred last chunk's fused cast), ``"fused_ip"`` (two or more chunks: the
    fused cast with the FP32 accumulator split at :func:`dw_split_rows`, written into the lower accumulator's own
    storage in place, :attr:`Plan.dw_split`), ``"unfused"`` (every chunk accumulated
    into the FP32 ``[V, H]`` accumulator, one flat cast after ``dX`` is written), ``"partial_ip"`` (two or more chunks:
    ``"fused_ip"``'s chunk loop and in-place cast over the vocabulary rows ``[0, dw_rows)``, ``dw_rows`` at least the
    split row; the recompute of the other rows covers the chunks before the last, whose ``dz`` stays pinned -- since
    the loss entry's forward, through the log-probability backward's chunk loop -- :attr:`Plan.dw_split_end`),
    ``"partial"`` (the chunk loop accumulates the vocabulary rows
    ``[0, dw_rows)``; the backward recomputes the others) or ``"recompute"`` (the backward recomputes every row).  ``dw_rows``: the vocabulary rows the chunk loop -- the loss entry's forward, the
    log-probability entry's backward -- accumulates (``V`` for the first two forms).  ``block_rows``: the rows of one
    vocabulary block of the backward's recompute (``None`` = no recompute; :func:`_dw_vocab_blocks`).
    ``predicted_bytes``: the predicted step peak above the caller's tensors (outputs included), in allocator blocks
    (:func:`_allocated_bytes`); ``budget_bytes``: the bound it was picked against."""

    form: str
    dw_rows: int
    block_rows: Optional[int]
    predicted_bytes: int
    budget_bytes: int

    @property
    def recompute(self) -> bool:
        """The backward recomputes the vocabulary rows ``[dw_rows, V)`` of the weight gradient."""
        return self.block_rows is not None

    @property
    def fits(self) -> bool:
        return self.predicted_bytes <= self.budget_bytes


MEMORY_FORMS = ("fused", "fused_ip", "unfused", "partial_ip", "partial", "recompute")


def memory_plan(
    entry: str,
    *,
    num_rows: int,
    rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    need_dx: bool = True,
    compact: bool = True,
    fuse_dw_cast: bool = True,
    recompute: bool = True,
    dx_ws_slabs: int = 0,
    buffer_rows: Optional[int] = None,
    x_copy: bool = False,
    dx_finalize: bool = True,
    budget_rows: Optional[int] = None,
    zero_rows: int = 0,
    zero_total: Optional[int] = None,
    exact_loss: bool = False,
    geometry: Geometry = DEFAULT_GEOMETRY,
    keep_dx_acc: bool = False,
    hidden_count: bool = False,
    zero_pass: str = "first",
) -> MemoryPlan:
    """The ``memory_mode="budget"`` rule: the fastest exact weight-gradient form whose predicted step peak stays within
    the budget -- ``(1 - MEMORY_BUDGET_MARGIN)`` times the bytes of a head that materializes its logits,
    ``2 * T_v * V + 2 * V * H + 2 * T * H`` (the BF16 logits of the ``T_v = budget_rows`` valid rows, the BF16 ``dW``
    and the BF16 ``dX``; the last term without ``dX``).  ``budget_rows`` (default ``rows``) is the number of valid rows
    whatever rows the chunk loop runs over: ``rows`` when compacted, while an uncompacted call (``rows = T``) still
    budgets the logits of its valid rows only; the model below counts the ``rows`` the chunk loop processes.

    Candidates in latency order: ``"fused"`` (when ``fuse_dw_cast`` allows it), ``"fused_ip"`` (a multi-chunk call
    with ``fuse_dw_cast``, either entry: the fused cast's chunk loop with the accumulator split at
    :func:`dw_split_rows`, whose backward writes ``dW`` into the lower accumulator's storage in the pieces of
    :func:`_dw_split_pieces` -- the ``dW`` phase holds the two accumulators, the deferred chunk and the first piece's
    temporary, no ``[V, H]`` output beside them), ``"unfused"``, then, when the record
    registers the recompute's logits arm (``recompute``), ``"partial_ip"`` (a multi-chunk call with ``fuse_dw_cast``,
    either entry: ``"fused_ip"`` over the rows ``[0, dw_rows)``, ``dw_rows`` at least :func:`dw_split_rows`, with
    the largest ``dw_rows``; its backward recomputes the rows ``[dw_rows, V)`` of the chunks before the last, the
    deferred chunk's pinned ``dz`` serving the last) and ``"partial"`` with the largest ``dw_rows`` -- the one that
    recomputes fewer chunk-rows of the vocabulary first -- and ``"recompute"``, each with the largest vocabulary block
    that fits; ``dw_rows`` and the block are multiples of :func:`dw_block_step`.  When no candidate fits (the chunk buffers of the ``chunk_size``, the dX cast, the copy of ``X``
    or the allocator's per-tensor bound of a small head leave no room), the one with the smallest predicted peak: the
    call runs it and its step peak can exceed the budget.  Every form is exact: the same ``dW`` bits
    (:func:`_dw_vocab_blocks`).

    The model counts, per phase, the buffers the entry points allocate -- the FP32 accumulators (``dX`` of the
    ``rows``, ``dW`` of the accumulated vocabulary rows), the chunk buffers of ``buffer_rows`` rows (logits,
    statistics, the gather buffer when ``compact``, ``dx_ws_slabs`` K-slice slabs), the outputs and a recompute block
    -- plus an estimate of the ``O(T)`` row vectors, and assumes a backward that frees the graph (the early release of
    the autograd entries; ``retain_graph=True`` keeps the forward's state).  Each counted tensor carries the default
    caching allocator's bound (``PYTORCH_CUDA_ALLOC_CONF`` unset; :func:`_allocated_bytes`), so a phase's prediction
    is its tensors' bytes plus 1 MiB per live tensor of 1 MiB or more and 512 bytes per live tensor.  Loss entry:
    forward, forward-to-backward state plus the ``dX`` output, then the ``dW`` phase; log-probability entry: its
    backward (the forward holds the chunk buffers only) -- the chunk loop, the ``dX`` output and, for the split
    accumulator, the cast and recompute phases beside the deferred chunk's ``dz`` (its chunk buffer) and ``X`` rows
    (gathered again when compacted).  ``buffer_rows`` defaults to ``min(chunk, rows)``: an
    explicit budget call runs without the hidden valid-row count, whose chunk-0 buffers take ``min(chunk, T)`` rows
    before the count is known; a default call that keeps the hidden count plans with ``buffer_rows=chunk`` (the count
    runs on calls of three chunks or more), ``compact=True`` (its gather buffer exists also when every row is valid)
    and ``hidden_count=True``.  ``x_copy`` (``Problem.x_copy``: ``X``'s leading stride or base is not a TMA pitch):
    an uncompacted call's chunk loop reads a contiguous BF16 ``[T, H]`` copy of ``X``, counted while it lives -- the
    loss entry's forward
    (and, in the fused form, until the backward's deferred chunk, whose ``X`` rows are a view of it) and the
    log-probability backward's chunk loop and casts.  ``dx_finalize`` (the call's resolved switch): a compacted call
    with the fused dX finalize writes ``dX`` from the FP32 accumulator; otherwise the dX is first cast to a BF16
    ``[rows, H]`` tensor -- the log-probability backward allocates it before its chunk loop, a compacted loss backward
    casts its rows before scattering them into the ``[T, H]`` output.

    ``zero_rows`` (loss entry): the zero-weight policy rows of a call whose chunk loop runs over its gradient rows only
    (``rows``), the ``budget_rows`` still counting every valid row; their logp-only pass runs before the chunk loop --
    a budget call runs without the hidden valid-row count, so no chunk buffer exists yet -- (its chunk buffers of
    ``min(chunk, zero_rows)`` rows, the gather buffer, the compact row vectors and the ``O(T)`` vectors) and is one
    more phase of the peak (``0``: no such pass).  ``zero_total`` (default ``zero_rows``): the call's zero-weight rows
    whether their pass runs or not (a call without ``logp`` skips it); ``budget_rows`` is at most
    ``rows + zero_total`` and ``rows + zero_total`` at most ``num_rows``.  ``exact_loss`` (a split call with the exact
    loss, :func:`.cake_backend._zero_weight_exact_loss`): after the chunk loop, with its accumulators and the deferred
    chunk held, the exact loss's buffers are one more phase (:func:`_exact_loss_bytes`).

    The fast form's own lifetimes (:func:`fast_form_fits`, the automatic form's check; the budget forms keep the
    defaults): ``keep_dx_acc`` (loss entry) -- the backward keeps the FP32 dX accumulator until it ends (the forms
    above release it once ``dX`` is written), so the ``dW`` phases hold it too; ``hidden_count`` (loss entry; also a
    budget form of a default call that keeps the count) -- the call runs with the hidden valid-row count, whose
    chunk-0 buffers (logits, statistics and gather buffer of ``buffer_rows`` rows) exist beside a zero-weight pass
    ahead of the loop and beside the rest of the chunk workspace a call that misses the binding cache allocates to
    prepare its binding on views of them (:func:`.cake_backend.workspace_layout` of ``min(chunk, rows)`` rows without
    its chunk buffers, :func:`.cake_backend._chunk_view_layout`; one more phase); ``zero_pass`` --
    where the zero-weight rows' pass runs
    (:func:`.cake_backend.zero_weight_memory`'s order): ``"first"`` ahead of the loop on its own chunk buffers,
    ``"arena"`` ahead of the loop on chunk buffers carved out of the loop's FP32 ``[V, H]`` accumulator (allocated
    ahead of it), ``"after"`` after the loop, beside the state the loop keeps for the backward."""
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}")
    if zero_pass not in ("first", "arena", "after"):
        raise ValueError(
            f"memory_plan's zero_pass must be 'first', 'arena' or 'after', got {zero_pass!r}"
        )
    T, R, H, V, C = (int(num_rows), int(rows), int(hidden), int(vocab), int(chunk))
    if R < 1 or T < R or H < 1 or V < 1 or C < 1:
        raise ValueError(
            f"memory_plan needs 1 <= rows <= num_rows and a positive geometry, got T={T} rows={R} H={H} V={V} C={C}"
        )
    Rb = R if budget_rows is None else int(budget_rows)
    Z = int(zero_rows)
    Zt = Z if zero_total is None else int(zero_total)
    if not 0 <= Rb <= R + Zt or not 0 <= Z <= Zt or R + Zt > T:
        raise ValueError(
            "memory_plan needs 0 <= budget_rows <= rows + zero_total, 0 <= zero_rows <= zero_total and "
            f"rows + zero_total <= num_rows, got budget_rows={Rb} rows={R} zero_rows={Z} zero_total={Zt} num_rows={T}"
        )
    step = dw_block_step(geometry)
    tile = int(geometry.stats_tile)
    n = -(-R // C)
    r_last = R - (n - 1) * C
    # ``"fused_ip"``: the accumulator's split row and the rows of the in-place cast's first piece
    split = dw_split_rows(V, geometry)
    piece0 = _dw_split_pieces(split, step)[0][1] if split < V else 0
    Cr = min(C, R)  # rows of the largest chunk
    Cb = Cr if buffer_rows is None else int(buffer_rows)
    live = _allocated_bytes  # every term below: its tensors with the allocator's per-tensor bound
    dxacc = live(4 * R * H) if need_dx else 0
    dx_out = live(2 * T * H) if need_dx else 0
    dw_out = live(2 * V * H)
    rowvec = (
        live(4 * T, 4) + live(4 * R, 8)
    )  # the O(T) vectors (16 T + 32 R bytes): labels, mask, scan, index, lse, logp, d, ...
    xc = (
        live(2 * T * H) if x_copy and not compact else 0
    )  # the contiguous copy of X the chunk loop reads
    dx_rows = (
        live(2 * R * H) if need_dx and not (compact and dx_finalize) else 0
    )  # the BF16 [rows, H] dX cast
    logits = live(2 * Cb * V)
    bufs = (
        logits
        + live(8 * Cb * -(-V // tile))
        + (live(2 * Cb * H) if compact else 0)
        + live(4 * int(dx_ws_slabs) * Cb * H)
    )
    # the hidden valid-row count's chunk-0 buffers (logits, statistics, gather buffer of ``buffer_rows`` rows), alive
    # from before the count is known until the chunk loop takes them over
    chunk0 = (
        logits + live(8 * Cb * -(-V // tile)) + live(2 * Cb * H) if hidden_count else 0
    )
    # a loss call that misses the binding cache prepares its binding while the hidden count's chunk-0 buffers are
    # alive: its template's logits, statistics and gather regions are views of those buffers, the other regions of the
    # chunk workspace of ``min(chunk, rows)`` rows (:func:`.cake_backend.workspace_layout`: row terms, K-slice slabs)
    # one allocation (``_chunk_views`` of :func:`prepare_lm_head_det_loss`) -- a phase of its own.  The accumulators
    # and outputs the template aliases into that allocation are no larger than the forward phase's, which holds them
    # beside chunk buffers of the same ``buffer_rows``.
    miss_phase = 0
    if entry == "loss" and hidden_count:
        workspace = _chunk_view_layout(
            workspace_layout(
                Cr,
                V,
                C,
                stats_tile=tile,
                dx_ws_slabs=int(dx_ws_slabs),
                hidden=H,
                compact=compact,
            )
        )["total"]
        miss_phase = chunk0 + live(workspace) + rowvec
    # the zero-weight rows' pass ahead of the chunk loop: a phase of its own
    zero_phase = 0
    zero_held = 0  # the pass after the chunk loop: beside the state the loop keeps for the backward
    if entry == "loss" and Z:
        # the zero-weight rows' logp-only pass: its chunk buffers (logits, statistics, gather buffer), its compact row
        # vectors (index, labels, lse, logp) and the call's O(T) vectors; ahead of the loop beside the hidden count's
        # chunk-0 buffers when it runs
        Cz = min(C, Z)
        zero_bufs = live(2 * Cz * V) + live(8 * Cz * -(-V // tile)) + live(2 * Cz * H)
        zero_rowvec = live(8 * Z, 2) + live(4 * Z, 2)
        if zero_pass == "after":
            zero_held = zero_bufs + zero_rowvec
        elif zero_pass == "arena":
            zero_phase = chunk0 + live(4 * V * H) + zero_rowvec + rowvec
        else:
            zero_phase = chunk0 + zero_bufs + zero_rowvec + rowvec
    # the fast form's FP32 dX accumulator kept through the dW phases (loss entry)
    kept = dxacc if keep_dx_acc else 0
    # the exact loss of a split call (:func:`_exact_loss_bytes`)
    exact = (
        _exact_loss_bytes(R, Cb, zero_rows=Z, zero_total=Zt)
        if entry == "loss" and exact_loss and Zt
        else 0
    )
    head = 2 * Rb * V + 2 * V * H + (2 * T * H if need_dx else 0)
    budget = head - int(math.ceil(head * float(MEMORY_BUDGET_MARGIN)))

    def block(vb: int) -> int:
        """One recompute block: the gathered rows, the block's ``dz`` and statistics, its FP32 accumulator (a
        multi-chunk call) and the block-shifted labels (with their temporary)."""
        return (
            live(2 * Cr * H)
            + live(2 * Cr * vb)
            + live(8 * Cr * -(-vb // tile))
            + (live(4 * vb * H) if n > 1 else 0)
            + live(8 * R)
            + live(4 * R)
        )

    if entry == "loss":
        x_last = (
            live(2 * r_last * H) if compact else 0
        )  # the deferred chunk's X rows, gathered again in the backward

        def tail_ip(vb: int) -> int:
            """``"partial_ip"``'s recompute of the rows ``[v0, V)``: ``dW`` on the lower accumulator's storage, the
            deferred chunk (its ``dz`` columns feed each block's cast) and one block."""
            return (
                dx_out
                + live(4 * split * H)
                + logits
                + x_last
                + block(vb)
                + rowvec
                + live(4 * R)
                + xc
                + kept
            )

        def peak(form: str, v0: int, vb: int) -> int:
            if form in ("fused", "fused_ip"):
                if n == 1:
                    acc = 0
                elif form == "fused":
                    acc = live(4 * V * H)
                else:  # the split accumulator: two allocations
                    acc = live(4 * split * H) + live(4 * (V - split) * H)
                fwd = dxacc + acc + bufs + rowvec + xc
                held = (
                    dxacc + acc + logits + rowvec + xc
                )  # the deferred chunk's dz pins its buffer, its X rows (a view) the copy
                last = (
                    dx_out
                    + acc
                    + logits
                    + x_last
                    # ``"fused_ip"``: ``dW`` takes the lower accumulator's storage; the first piece's temporary
                    + (dw_out if form == "fused" else live(2 * piece0 * H))
                    + rowvec
                    + xc
                    + kept
                )
                return max(
                    fwd,
                    held + max(exact, zero_held),
                    held + dx_out + live(4 * T) + (dx_rows if compact else 0),
                    last,
                )
            if form == "partial_ip":
                # the split accumulator of the rows [0, v0) (lower [0, split), upper [split, v0)), the deferred chunk
                # pinned as in the fused forms, the saved row scale for the tail's recompute
                lower = live(4 * split * H)
                upper = live(4 * (v0 - split) * H) if v0 > split else 0
                keep = live(4 * R)
                held = dxacc + lower + upper + logits + rowvec + keep + xc
                out = max(
                    dxacc + lower + upper + bufs + rowvec + keep + xc,
                    held + max(exact, zero_held),
                    held + dx_out + live(4 * T) + (dx_rows if compact else 0),
                    # the in-place cast of [0, v0): both accumulators, the deferred chunk, the first piece's temporary
                    dx_out
                    + lower
                    + upper
                    + logits
                    + x_last
                    + live(2 * piece0 * H)
                    + rowvec
                    + keep
                    + xc
                    + kept,
                )
                if vb:
                    out = max(out, tail_ip(vb))
                return out
            acc = live(4 * v0 * H)
            keep = (
                live(4 * R) if vb else 0
            )  # the per-row gradient scale of the recompute
            out = max(
                dxacc + acc + bufs + rowvec + keep + xc,
                dxacc + acc + rowvec + keep + xc + max(exact, zero_held),
                dxacc
                + acc
                + rowvec
                + keep
                + dx_out
                + live(4 * T)
                + (dx_rows if compact else 0),
            )
            if v0:
                out = max(out, dx_out + dw_out + acc + rowvec + keep + kept)
            if vb:
                out = max(out, dx_out + dw_out + block(vb) + rowvec + keep + kept)
            return out

    else:
        loop = (
            bufs + live(4 * R) + rowvec + xc + dx_rows
        )  # the backward's chunk buffers, compact dlogp and dX cast
        x_last = (
            live(2 * r_last * H) if compact else 0
        )  # the deferred chunk's X rows, gathered again for the split accumulator's cast

        def tail_ip(vb: int) -> int:
            """``"partial_ip"``'s recompute of the rows ``[v0, V)``: ``dW`` on the lower accumulator's storage, the
            deferred chunk (its ``dz`` columns feed each block's cast), one block and the compact ``dlogp``."""
            return (
                dx_out
                + live(4 * split * H)
                + logits
                + x_last
                + block(vb)
                + rowvec
                + live(4 * R)
                + xc
            )

        def peak(form: str, v0: int, vb: int) -> int:
            if form == "fused":
                acc = live(4 * V * H) if n > 1 else 0
                return max(
                    dxacc + acc + dw_out + loop,
                    dxacc + dx_out + dw_out + rowvec + live(4 * T) + xc,
                )
            if form in ("fused_ip", "partial_ip"):
                # the split accumulator of the rows [0, v0) (lower [0, split), upper [split, v0)); after the chunk
                # loop the deferred chunk's dz pins its chunk buffer until the cast and the recompute have read it
                lower = live(4 * split * H)
                upper = live(4 * (v0 - split) * H) if v0 > split else 0
                out = max(
                    dxacc + lower + upper + loop,
                    dxacc + dx_out + lower + upper + logits + rowvec + live(4 * T) + xc,
                    # the in-place cast of [0, v0): both accumulators, the deferred chunk, the first piece's temporary
                    dx_out
                    + lower
                    + upper
                    + logits
                    + x_last
                    + live(2 * piece0 * H)
                    + rowvec
                    + xc,
                )
                if vb:
                    out = max(out, tail_ip(vb))
                return out
            acc = live(4 * v0 * H)
            out = max(
                dxacc + acc + loop, dxacc + dx_out + acc + rowvec + live(4 * T) + xc
            )
            if v0:
                out = max(out, dx_out + dw_out + acc + rowvec)
            if vb:
                out = max(out, dx_out + dw_out + block(vb) + rowvec + live(4 * R))
            return out

    def largest(fits, hi: int) -> int:
        """The largest multiple of ``step`` in ``[step, hi]`` with ``fits`` (monotone), else 0."""
        lo_k, hi_k = 0, hi // step
        while lo_k < hi_k:
            mid = (lo_k + hi_k + 1) // 2
            if fits(mid * step):
                lo_k = mid
            else:
                hi_k = mid - 1
        return lo_k * step

    candidates: list[tuple[str, int, Optional[int]]] = []
    if fuse_dw_cast:
        candidates.append(("fused", V, None))
        if n > 1 and split < V:
            candidates.append(("fused_ip", V, None))
    candidates.append(("unfused", V, None))
    if recompute and step < V:
        # the largest block whose phase fits (it does not depend on the accumulated rows), at least one granule
        vb_max = largest(lambda vb: peak("recompute", 0, vb) <= budget, V) or step
        # the largest accumulated prefix whose phases fit with the block it leaves (a block within vb_max fits, so the
        # predicate falls monotonically with the prefix); the form's own peak, saved row scale included
        v0 = largest(
            lambda v: peak("partial", v, min(vb_max, V - v)) <= budget, V - step
        )
        partial = [("partial", v0, min(vb_max, V - v0))] if v0 else []
        ip = []
        if fuse_dw_cast and n > 1 and step <= split < V:
            # ``"partial_ip"``: the largest tail block (its phase does not depend on the accumulated rows), then the
            # largest accumulated end ``v0 >= split`` whose phases fit with the block it leaves
            vb_ip = largest(lambda vb: tail_ip(vb) <= budget, V - split)
            if vb_ip:
                k = largest(
                    lambda v: peak(
                        "partial_ip", split + v - step, min(vb_ip, V - split - v + step)
                    )
                    <= budget,
                    V - split,
                )
                if k:
                    v_ip = split + k - step
                    ip = [("partial_ip", v_ip, min(vb_ip, V - v_ip))]
        # recompute work: ``"partial"`` recomputes the rows ``[v0, V)`` of every chunk, ``"partial_ip"`` those of the
        # chunks before the last (the last one's ``dz`` stays pinned); the one with less goes first
        if ip and partial and (n - 1) * (V - ip[0][1]) > n * (V - partial[0][1]):
            candidates += partial + ip
        else:
            candidates += ip + partial
        candidates.append(("recompute", 0, vb_max))
    scored = [
        (form, v0, vb, max(peak(form, v0, vb or 0), zero_phase, miss_phase))
        for form, v0, vb in candidates
    ]
    chosen = next(
        (c for c in scored if c[3] <= budget), min(scored, key=lambda c: c[3])
    )
    form, v0, vb, predicted = chosen
    return MemoryPlan(
        form=form,
        dw_rows=int(v0),
        block_rows=None if vb is None else int(vb),
        predicted_bytes=int(predicted),
        budget_bytes=int(budget),
    )


def fast_form_fits(entry: str, *, fuse_dw_cast: bool = True, **kw: Any) -> bool:
    """Whether the fast form -- the fused cast (``fuse_dw_cast``) or the unfused one, without a recompute -- fits the
    budget of :func:`memory_plan` with its own buffer lifetimes: the loss entry's backward keeps the FP32 dX
    accumulator through the weight gradient (``keep_dx_acc``; the log-probability phases hold it already).  ``kw``:
    :func:`memory_plan`'s other keywords (``hidden_count`` / ``zero_pass``: where the fast form's zero-weight pass
    runs).  The check is ``fits`` and the fast form itself: when no form fits, :func:`memory_plan` returns the one with
    the smallest predicted peak, which can be the fused one.  Pure host arithmetic."""
    fast = memory_plan(
        entry,
        fuse_dw_cast=fuse_dw_cast,
        recompute=False,
        keep_dx_acc=True,
        **kw,
    )
    return fast.fits and fast.form == ("fused" if fuse_dw_cast else "unfused")


def _auto_plans(rows: int, chunk: int) -> bool:
    """Whether the automatic form plans a call whose fast form chunks over ``rows`` rows: at least
    :data:`_AUTO_MIN_CHUNKS` chunks of ``chunk`` rows (a shorter call keeps the fast form without planning)."""
    rows = int(rows)
    return rows > 0 and -(-rows // int(chunk)) >= _AUTO_MIN_CHUNKS


def auto_memory_plan(
    entry: str, *, fuse_dw_cast: bool = True, recompute: bool = True, **kw: Any
) -> Optional[MemoryPlan]:
    """The automatic form of a default call (:data:`MEMORY_AUTO`): ``None`` -- the fast form -- when the call runs
    fewer than :data:`_AUTO_MIN_CHUNKS` chunks, when the fast form fits the budget with its own lifetimes
    (:func:`fast_form_fits`), or when no budget form fits either (the fast form then, rather than the recompute form of
    the smallest predicted peak); else the budget form :func:`memory_plan` picks (``"budget"``'s form, which fits).
    ``kw``: :func:`memory_plan`'s keywords of the call, shared by both checks.  Pure host arithmetic on integers: it
    reads neither the allocator's state nor the device's free memory nor an environment variable, so the forward and a
    checkpoint's recomputation pick the same form."""
    if not _auto_plans(kw["rows"], kw["chunk"]):
        return None
    if fast_form_fits(entry, fuse_dw_cast=fuse_dw_cast, **kw):
        return None
    plan = memory_plan(entry, fuse_dw_cast=fuse_dw_cast, recompute=recompute, **kw)
    return plan if plan.fits else None


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
    # (``int``: an :class:`_UpperRows` index is chunk ``index``'s, though never equal to the plain index)
    first, last = int(int(index) == 0), int(int(index) == plan.num_chunks - 1)
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
        # a ``dw_rows`` plan: the GEMM's M is the vocabulary rows [0, dw_rows), A the chunk's dz columns of them; a
        # ``dw_split`` plan: [0, dw_split) into ``dw_acc``, [dw_split, V) into ``dw_acc_hi`` (the :class:`_UpperRows`
        # launch; A's base the column offset)
        v0, v1 = plan.dw_acc_rows(index)
        values.update(
            A=dz_chunk if (v0, v1) == (0, p.vocab) else dz_chunk[:, v0:v1],
            B=x_chunk,
            C=t["dw_acc_hi"] if isinstance(index, _UpperRows) else t["dw_acc"],
            STATS_OUT=t["stats"],
            WS=t["f32_dummy"],
            **gemm_scalars(
                stage,
                g,
                hidden=p.hidden,
                vocab=v1 - v0,
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
    elif stage == "save_row_grad":
        # the budget forms' recompute reads every chunk's per-row gradient scale in the backward
        values.update(src=t["d"][:rows_c], out=t["d_rows"][row0:stop])
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
    return values


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
        v0, v1 = plan.dw_acc_rows(index)
        values.update(
            A=logits[:rows_c] if (v0, v1) == (0, p.vocab) else logits[:rows_c, v0:v1],
            B=x_chunk,
            C=t["dw_acc_hi"] if isinstance(index, _UpperRows) else t["dw_acc"],
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
    return values


def _slot_values_temperature(
    stage: str, t: dict[str, Any], plan: Plan, index
) -> dict[str, Any]:
    """:func:`_slot_values` of a binding with the temperature forms (:class:`_TemperatureBinding`): the role's views
    (``row_finalize_tau`` is ``row_finalize``'s temperature form) plus the compact ``temperature`` rows.  The tensor
    host values of :func:`stage_values` for chunk ``index`` -- the views a remembered launch slots in per
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
    base = (
        base_stage(stage)
        if (stage in GEMM_STAGES or stage in TEMPERATURE_STAGES)
        else stage
    )
    tau_rows = t["temperature"] if p.temperature else None
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
        if tau_rows is not None:
            values.update(temperature=tau_rows)
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
        # a ``dw_rows`` / ``dw_split`` plan: the accumulated vocabulary rows (:meth:`Plan.dw_acc_rows`; the
        # :class:`_UpperRows` launch into ``dw_acc_hi``)
        v0, v1 = plan.dw_acc_rows(index)
        values.update(
            A=logits[:rows_c] if (v0, v1) == (0, p.vocab) else logits[:rows_c, v0:v1],
            B=x_chunk,
            C=t["dw_acc_hi"] if isinstance(index, _UpperRows) else t["dw_acc"],
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
    elif base == "row_finalize":
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
        if tau_rows is not None:
            values.update(temperature=tau_rows)
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
        if tau_rows is not None:
            values.update(temperature=tau_rows)
    else:
        return stage_values(stage, t, plan, index)
    return values


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
        if plan.keep_row_grad:
            keys.append(
                ("save_row_grad", index)
            )  # the chunk's ``d`` into ``d_rows`` before the next chunk's row_finalize reuses ``d``
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
                if plan.dw_split_upper:
                    # the split accumulator's upper rows: the same form again (:func:`_run_split_keys` runs it right
                    # behind the first, on the same stream)
                    keys.append((plan.dw_acc_stage(index), _UpperRows(index)))
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
        if plan.keep_row_grad:
            keys.append(
                ("save_row_grad", index)
            )  # the chunk's ``d`` into ``d_rows`` before the next chunk's row_finalize reuses ``d``
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


def _run_split_keys(
    plan: Plan,
    device_index: int,
    keys: tuple,
    run_key: Callable[[Any], None],
    dw_stream: Optional[Union[bool, str]] = None,
    run_keys: Optional[Callable[..., None]] = None,
) -> None:
    """``run_keys`` (default :func:`.cake_backend._run_keys`, looked up at the call; ``_run_keys_temperature`` for a plan
    with the temperature forms) of a plan with a split weight-gradient accumulator (:attr:`Plan.dw_split`; any other
    plan: that function).  Each chunk's second accumulate (its :class:`_UpperRows` key) runs right behind the first, on
    the same stream -- the dW side stream where the placement of :func:`side_stream_schedule` puts the chunk's
    accumulate -- so that placement sees one accumulate per chunk, as without the split."""
    if run_keys is None:
        run_keys = _run_keys
    if plan.dw_split is None:
        run_keys(plan, device_index, keys, run_key, dw_stream)
        return
    upper = {key[1]: key for key in keys if isinstance(key[1], _UpperRows)}

    def run_pair(key) -> None:
        run_key(key)
        pair = upper.get(_UpperRows(key[1])) if type(key[1]) is int else None
        if (
            pair is not None and pair[0] == key[0]
        ):  # the chunk's first accumulate: its second right behind
            run_key(pair)

    run_keys(
        plan,
        device_index,
        tuple(key for key in keys if not isinstance(key[1], _UpperRows)),
        run_pair,
        dw_stream,
    )


def cast_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys of the output casts: the flat ``dX`` cast and, for ``dW``, the flat cast of the accumulator or -- when
    the plan defers its last chunk -- that chunk's fused GEMM (its ``X`` rows gathered again first when the rows are
    compacted; the log-probability backward runs it inside :func:`recompute_keys` instead)."""
    keys: list[tuple[str, Any]] = []
    if plan.need_dx and plan.dx_cast:
        keys.append(("scale_cast_bf16", "dx"))
    if (
        plan.need_dw and plan.dw_rows is None
    ):  # a ``dw_rows`` plan leaves its accumulator to the caller
        if not plan.dw_deferred:
            keys.append((plan.dw_cast_stage, "dw"))
        elif (
            plan.problem.entry == "loss" and plan.dw_split is None
        ):  # a ``dw_split`` plan leaves its cast to the caller (:func:`_dw_cast_split`)
            last = plan.num_chunks - 1
            if plan.compact:
                keys.append(("gather_rows", last))
            keys.append((plan.dw_cast_stage, last))
    return tuple(keys)


def _with_split_keys(plan: Plan, keys: tuple) -> tuple[tuple[str, Any], ...]:
    """The log-probability backward's launch keys (:func:`.cake_backend.recompute_keys` or its temperature form) of a
    plan with a split weight-gradient accumulator (:attr:`Plan.dw_split`; any other plan: ``keys``): each chunk's
    accumulate is followed by its :class:`_UpperRows` key when the accumulator has an upper part, and the deferred last
    chunk runs no weight-gradient GEMM -- its ``dz`` stays in the chunk buffer and the caller casts ``dW`` from it
    (:func:`_dw_cast_split`)."""
    if plan.dw_split is None:
        return keys
    last = plan.num_chunks - 1
    out: list[tuple[str, Any]] = []
    for stage, index in keys:
        if type(index) is int and index == last and stage == plan.dw_cast_stage:
            continue  # the deferred chunk's fused cast: the caller's (:func:`_dw_cast_split`)
        out.append((stage, index))
        if (
            plan.dw_split_upper
            and type(index) is int
            and stage == plan.dw_acc_stage(index)
        ):
            out.append((stage, _UpperRows(index)))
    return tuple(out)


class _SplitDw(NamedTuple):
    """The weight-gradient state a log-probability backward with a split accumulator (:attr:`Plan.dw_split`) leaves
    to its caller: ``acc`` = ``[lower, upper]`` (the FP32 sums of the vocabulary rows ``[0, dw_split)`` and
    ``[dw_split, dw_split_end)`` over the chunks before the last; ``upper`` ``None`` when empty) and the deferred last
    chunk's operands of :func:`.cake_backend._deferred_operands` -- its BF16 ``dz`` rows (a view of the chunk buffer)
    and ``X`` rows (``x_last``; or ``x_src`` / ``x_idx`` to gather when compacted)."""

    acc: list
    dz_last: torch.Tensor
    x_last: Optional[torch.Tensor]
    x_src: Optional[torch.Tensor]
    x_idx: Optional[torch.Tensor]

    def x_rows(self) -> torch.Tensor:
        """The deferred chunk's BF16 ``X`` rows (gathered when compacted)."""
        if self.x_last is not None:
            return self.x_last
        return torch.index_select(self.x_src, 0, self.x_idx)


def _logprob_dw_result(plan: Plan, t: dict[str, Any]) -> Any:
    """The weight-gradient result of a log-probability backward over the bound tensors ``t``: the BF16 ``dw_out``, a
    ``dw_rows`` plan's FP32 accumulator, a ``dw_split`` plan's :class:`_SplitDw`."""
    if plan.dw_split is not None and plan.need_dw:
        return _SplitDw([t["dw_acc"], t.get("dw_acc_hi")], *_deferred_operands(plan, t))
    return t.get("dw_out") if plan.dw_rows is None else t.get("dw_acc")


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
    # built only to template a remembered binding (``prepare_lm_head_det_loss(_template_only=True)``): its buffers alias
    # one another, so it never launches
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
                "this runner only templates a remembered binding (its buffers alias one another); it never launches"
            )
        if self.backend == "cake":
            self.prepare_tma()

            def run_key(key) -> None:
                host = HOST_STAGES.get(key[0])
                if host is not None:
                    host(self.values[key])
                else:
                    self.launches[key]()

            _run_split_keys(self.plan, self.device_index, keys, run_key, self.dw_stream)
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
        # a ``dw_rows`` plan: the caller casts the FP32 accumulator of the rows it covers; a log-probability
        # ``dw_split`` plan: the caller casts from the split accumulator and the deferred chunk (:class:`_SplitDw`)
        if self.problem.entry == "logprob":
            return dx, _logprob_dw_result(self.plan, t)
        return dx, (t.get("dw_out") if self.plan.dw_rows is None else t.get("dw_acc"))

    def step(self, grad: Optional[torch.Tensor] = None):
        self.forward()
        return self.backward(grad)

    __call__ = step


class _TemperatureRunner(LmHeadDetLossRunner):
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

        _run_split_keys(
            self.plan,
            self.device_index,
            keys,
            run_key,
            self.dw_stream,
            run_keys=_run_keys_temperature,
        )


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
    dw_rows: Optional[int] = None,
    keep_row_grad: bool = False,
    _template_only: bool = False,
    dx_slices_cap: Optional[int] = None,
    _chunk_views: Optional[_HiddenCount] = None,
    dw_split: Optional[int] = None,
    dw_split_end: Optional[int] = None,
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

    ``dw_rows`` / ``keep_row_grad`` (the budget forms of :func:`memory_plan`; :attr:`Plan.dw_rows`,
    :attr:`Plan.keep_row_grad`): the weight-gradient accumulate covers the vocabulary rows ``[0, dw_rows)`` of an
    unfused plan and ``backward()`` returns the FP32 ``[dw_rows, H]`` accumulator in place of ``dw_out`` (no cast);
    ``keep_row_grad`` keeps every chunk's per-row gradient scale in ``tensors["d_rows"]``.  ``dw_split`` (the budget
    form ``"fused_ip"``, :attr:`Plan.dw_split`): the chunk loop accumulates into ``dw_acc`` (FP32 ``[dw_split, H]``)
    and ``tensors["dw_acc_hi"]`` (FP32 ``[dw_split_end - dw_split, H]``, ``dw_split_end`` default ``V``; none when
    empty); the deferred last chunk's cast is left to the caller (:func:`_dw_cast_split`): ``dw_out`` is neither bound
    nor allocated and ``backward()`` returns no ``dW`` (a log-probability runner: the split accumulator and the
    deferred chunk, :class:`_SplitDw`).  ``dw_split_end`` (the budget form ``"partial_ip"``; the loss entry with
    ``keep_row_grad``): the upper part ends there; the caller recomputes the rows above it.

    ``_template_only`` (internal; the eager entry points, ``cake`` backend): the runner only templates a remembered
    binding (:meth:`_Binding.from_runner`), which launches with per-call buffers of its own.  The workspace and the
    accumulators / casts allocated here then share one allocation of the largest of them (views of it that alias one
    another), so building a binding costs the largest buffer rather than their sum; such a runner never launches.
    ``_chunk_views`` (internal; with ``_template_only``, the call's hidden valid-row count): the template's logits,
    statistics and gather regions are views of the count's chunk-0 buffers (:data:`.cake_backend.CHUNK_VIEW_REGIONS`)
    and the other regions share that one allocation (:func:`.cake_backend._chunk_view_layout`): a call that misses the
    binding cache prepares its binding beside the count's chunk buffers without a second chunk buffer.
    """
    if _template_only and backend != "cake":
        raise ValueError("a template-only runner templates a cake-backend binding")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if entry != "logprob" and (logprob_dz or logprob_lbl):
        raise ValueError(
            "logprob_dz / logprob_lbl select logits GEMM arms of the log-probability entry"
        )
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
    plan = make_plan(
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
        dw_rows=dw_rows,
        keep_row_grad=keep_row_grad,
        dx_slices_cap=None if dx_slices_cap is None else int(dx_slices_cap),
        dw_split=dw_split,
        dw_split_end=dw_split_end,
    )
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
    if _chunk_views is not None and (
        not _template_only or backend != "cake" or workspace_buffer is not None
    ):
        raise ValueError(
            "_chunk_views templates a cake-backend runner (_template_only) on its own workspace"
        )
    # a template on the hidden valid-row count's chunk-0 buffers: the other regions in one smaller allocation
    carved = layout if _chunk_views is None else _chunk_view_layout(layout)
    arena = None
    alias: dict[str, int] = {}
    if _template_only:
        # the buffers allocated below (never launched on): the workspace and the accumulators / casts the call leaves
        # to this function, each a view at the start of one allocation of the largest
        f32 = _main._dw_cast_f32(plan.dw_cast_stage)
        sizes = {
            "dx_acc": rows_loop * H * 4 if plan.need_dx and dx_acc is None else 0,
            "dx_out": rows_loop * H * 2
            if plan.need_dx and plan.dx_cast and dx_out is None
            else 0,
            "dw_acc": (plan.dw_acc_rows(0)[1] - plan.dw_acc_rows(0)[0]) * H * 4
            if plan.need_dw and plan.dw_acc_needed and dw_acc is None
            else 0,
            "dw_acc_hi": (
                plan.dw_acc_rows(_UpperRows(0))[1] - plan.dw_acc_rows(_UpperRows(0))[0]
            )
            * H
            * 4
            if plan.need_dw and plan.dw_split_upper
            else 0,
            "dw_out": V * H * (4 if f32 else 2)
            if plan.need_dw
            and plan.dw_rows is None
            and plan.dw_split is None
            and dw_out is None
            else 0,
        }
        alias = {name: n for name, n in sizes.items() if n}
        arena = torch.empty(
            max([carved["total"] if workspace_buffer is None else 0, *alias.values()]),
            dtype=torch.uint8,
            device=device,
        )
        if workspace_buffer is None:
            workspace_buffer = arena
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            layout["total"], dtype=torch.uint8, device=device
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
        if given is None and arena is not None and name in alias:
            return arena[: alias[name]].view(dtype).view(shape)
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
    if plan.need_dx:
        t["dx_acc"] = output(dx_acc, "dx_acc", (rows_loop, H), torch.float32)
        if plan.dx_cast:
            t["dx_out"] = output(dx_out, "dx_out", (rows_loop, H), torch.bfloat16)
    if plan.need_dw:
        if plan.dw_acc_needed:
            v0, v1 = plan.dw_acc_rows(0)
            t["dw_acc"] = output(dw_acc, "dw_acc", (v1 - v0, H), torch.float32)
        if plan.dw_split_upper:
            # the split accumulator's upper rows (a template's view aliases the arena like the other buffers)
            u0, u1 = plan.dw_acc_rows(_UpperRows(0))
            t["dw_acc_hi"] = output(None, "dw_acc_hi", (u1 - u0, H), torch.float32)
        if (
            plan.dw_rows is None and plan.dw_split is None
        ):  # a ``dw_rows`` / ``dw_split`` plan leaves its accumulator(s) to the caller
            t["dw_out"] = output(
                dw_out,
                "dw_out",
                (V, H),
                torch.float32
                if _main._dw_cast_f32(plan.dw_cast_stage)
                else torch.bfloat16,
            )
    if plan.keep_row_grad:
        t["d_rows"] = torch.empty((rows_loop,), dtype=torch.float32, device=device)
    if layout.get("workspace"):
        t["workspace"] = carve("workspace", torch.uint8, (layout["workspace"][1],))
    if layout.get("tma_descriptor_workspace"):
        t["tma_descriptor_workspace"] = carve(
            "tma_descriptor_workspace",
            torch.uint8,
            (layout["tma_descriptor_workspace"][1],),
        )

    fwd = forward_keys(plan)
    bwd = (
        _with_split_keys(plan, recompute_keys(plan)) if entry == "logprob" else ()
    ) + cast_keys(plan)
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
    if device.type == "cuda":
        device_index = int(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    else:
        device_index = 0
    return LmHeadDetLossRunner(
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
        _template_only=_template_only,
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
    dw_rows: Optional[int] = None,
    keep_row_grad: bool = False,
    temperature: Optional[torch.Tensor] = None,
    _template_only: bool = False,
    dw_split: Optional[int] = None,
    dw_split_end: Optional[int] = None,
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

    ``dw_rows`` / ``keep_row_grad`` (the budget forms of :func:`memory_plan`; :attr:`Plan.dw_rows`,
    :attr:`Plan.keep_row_grad`): the weight-gradient accumulate covers the vocabulary rows ``[0, dw_rows)`` of an
    unfused plan and ``backward()`` returns the FP32 ``[dw_rows, H]`` accumulator in place of ``dw_out`` (no cast);
    ``keep_row_grad`` keeps every chunk's per-row gradient scale in ``tensors["d_rows"]``.  ``dw_split`` /
    ``dw_split_end`` (the log-probability budget forms ``"fused_ip"`` / ``"partial_ip"``, :attr:`Plan.dw_split`): as
    in :func:`prepare_lm_head_det_loss`; ``backward()`` returns the split accumulator and the deferred chunk
    (:class:`_SplitDw`) in place of ``dw_out``.

    ``temperature`` (log-probability entry only; ``None`` = none): the contiguous FP32 per-row temperature of the
    chunk loop's rows -- like ``lse``, the compact ``[T_v]`` rows of the valid-row index when the runner is compacted,
    else ``[T]`` -- bound as ``tensors["temperature"]`` (new values may be written into it between steps).  Its values
    must be positive and finite; the runner does not check them (the eager entry points validate the caller's
    ``temperature``, :func:`_resolve_temperature`).  On the ``cake`` backend the plan runs the temperature forms
    (:data:`TEMPERATURE_STAGES`, both logits GEMM arms: ``logprob_dz`` / ``logprob_lbl`` must not be ``False``; a
    record without them raises ``NotImplementedError``); the reference backend reads it in the same stages.

    ``_template_only`` (internal; the eager entry points, ``cake`` backend): the runner only templates a remembered
    binding (:meth:`_Binding.from_runner`), which launches with per-call buffers of its own.  The workspace and the
    accumulators / casts allocated here then share one allocation of the largest of them (views of it that alias one
    another), so building a binding costs the largest buffer rather than their sum; such a runner never launches.
    """
    if _template_only and backend != "cake":
        raise ValueError("a template-only runner templates a cake-backend binding")
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
    plan = (
        make_plan_temperature
        if problem.temperature and record is not None
        else make_plan
    )(
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
        dw_rows=dw_rows,
        keep_row_grad=keep_row_grad,
        dw_split=dw_split,
        dw_split_end=dw_split_end,
    )
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
    arena = None
    alias: dict[str, int] = {}
    if _template_only:
        # the buffers allocated below (never launched on): the workspace and the accumulators / casts the call leaves
        # to this function, each a view at the start of one allocation of the largest
        f32 = _main._dw_cast_f32(plan.dw_cast_stage)
        sizes = {
            "dx_acc": rows_loop * H * 4 if plan.need_dx and dx_acc is None else 0,
            "dx_out": rows_loop * H * 2
            if plan.need_dx and plan.dx_cast and dx_out is None
            else 0,
            "dw_acc": (plan.dw_acc_rows(0)[1] - plan.dw_acc_rows(0)[0]) * H * 4
            if plan.need_dw and plan.dw_acc_needed and dw_acc is None
            else 0,
            "dw_acc_hi": (
                plan.dw_acc_rows(_UpperRows(0))[1] - plan.dw_acc_rows(_UpperRows(0))[0]
            )
            * H
            * 4
            if plan.need_dw and plan.dw_split_upper
            else 0,
            "dw_out": V * H * (4 if f32 else 2)
            if plan.need_dw
            and plan.dw_rows is None
            and plan.dw_split is None
            and dw_out is None
            else 0,
        }
        alias = {name: n for name, n in sizes.items() if n}
        arena = torch.empty(
            max([layout["total"] if workspace_buffer is None else 0, *alias.values()]),
            dtype=torch.uint8,
            device=device,
        )
        if workspace_buffer is None:
            workspace_buffer = arena
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
        if given is None and arena is not None and name in alias:
            return arena[: alias[name]].view(dtype).view(shape)
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
    )  # unused ``WS`` of the unsliced GEMMs
    if plan.dx_ws_slabs:
        t["dx_ws"] = _carve(
            flat, layout, "dx_ws", torch.float32, (plan.dx_ws_slabs, rows, H)
        )
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
    if plan.need_dx:
        t["dx_acc"] = output(dx_acc, "dx_acc", (rows_loop, H), torch.float32)
        if plan.dx_cast:
            t["dx_out"] = output(dx_out, "dx_out", (rows_loop, H), torch.bfloat16)
    if plan.need_dw:
        if plan.dw_acc_needed:
            v0, v1 = plan.dw_acc_rows(0)
            t["dw_acc"] = output(dw_acc, "dw_acc", (v1 - v0, H), torch.float32)
        if plan.dw_split_upper:
            # the split accumulator's upper rows (a template's view aliases the arena like the other buffers)
            u0, u1 = plan.dw_acc_rows(_UpperRows(0))
            t["dw_acc_hi"] = output(None, "dw_acc_hi", (u1 - u0, H), torch.float32)
        if (
            plan.dw_rows is None and plan.dw_split is None
        ):  # a ``dw_rows`` / ``dw_split`` plan leaves its accumulator(s) to the caller
            t["dw_out"] = output(
                dw_out,
                "dw_out",
                (V, H),
                torch.float32
                if _main._dw_cast_f32(plan.dw_cast_stage)
                else torch.bfloat16,
            )
    if plan.keep_row_grad:
        t["d_rows"] = torch.empty((rows_loop,), dtype=torch.float32, device=device)
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
    bwd = (
        _with_split_keys(plan, _recompute_keys_temperature(plan))
        if entry == "logprob"
        else ()
    ) + cast_keys(plan)
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
        _template_only=_template_only,
    )


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
    the resolved log-probability forward's logits form (:func:`logprob_lbl_default`).
    A budget form of :func:`memory_plan` appends its plan options ``(dw_rows, keep_row_grad)``, the forms
    ``"fused_ip"`` / ``"partial_ip"`` ``("dw_split", dw_split, dw_split_end, keep_row_grad)``."""
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
) -> tuple:
    """Cache key of a log-probability backward binding (as :func:`forward_binding_key`); ``logprob_dz`` the resolved
    recompute form (:func:`logprob_dz_default`).  A budget form appends its accumulated rows (:attr:`Plan.dw_rows`),
    the forms ``"fused_ip"`` / ``"partial_ip"`` ``("dw_split", dw_split, dw_split_end)``.
    ``lse`` and ``dlogp`` are keyed by layout alone (:func:`_layout`), as in :mod:`.cake_backend`."""
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
    )


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
            # fields of the base ``cake_backend.Plan`` (unseen when mypy skips the import)
            plan = replace(  # type: ignore[call-arg]
                plan,
                hidden_count=True,
                hidden_logits_group_m=hidden.group_m,
                hidden_stats=hidden.stats,
            )
            forward_order = forward_keys(plan)
            backward_order = (
                _with_split_keys(plan, recompute_keys(plan))
                if p.entry == "logprob"
                else ()
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
    ) -> None:
        """Per-call temporaries from the caching allocator into ``t`` (the runner's regions, minus the owned ones);
        with ``hidden`` the chunk buffers are the prologue's (chunk 0's logits / statistics / gathered rows already in
        them, ``min(chunk, T)`` rows) -- the entry points check that a hidden-count plan gets its prologue.
        ``chunk_buffers=False``: the chunk buffers are already bound (:meth:`_chunk_buffers`, the early launch of
        :meth:`forward`) and the rest is allocated."""
        p = self.plan.problem
        rows = self.layout["logits"][1] // (p.vocab * 2)
        if chunk_buffers and hidden is not None:
            t["logits"], t["stats"] = hidden.logits, hidden.stats_buf
        elif chunk_buffers:
            t["logits"] = torch.empty(
                (rows, p.vocab), dtype=torch.bfloat16, device=self.device
            )
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
        if self.plan.dx_ws_slabs:
            t["dx_ws"] = torch.empty(
                (self.plan.dx_ws_slabs, rows, p.hidden),
                dtype=torch.float32,
                device=self.device,
            )
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

    def _chunk_buffers(self, t: dict[str, Any]) -> None:
        """What chunk 0's row gather and logits GEMM bind (the early launch of :meth:`forward`): the chunk's logits,
        statistics and gather buffers of :meth:`_scratch` (the same allocations, ahead of the others), the constant
        cells and the owned regions."""
        p = self.plan.problem
        rows = self.layout["logits"][1] // (p.vocab * 2)
        t["logits"] = torch.empty(
            (rows, p.vocab), dtype=torch.bfloat16, device=self.device
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

        _run_split_keys(self.plan, self.device_index, keys, run_key, dw_stream)

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
        dw_acc: Optional[torch.Tensor] = None,
    ):
        """One forward through the remembered launches; ``dw_stream`` is the call's dW side-stream mode
        (:func:`dw_stream_mode`), a launch placement outside the binding.  ``capture`` (a loss forward of a zero-weight
        split with the exact loss, :func:`.cake_backend._zero_weight_exact_loss`): the chunk loop keeps its loss terms
        (:func:`.cake_backend._term_rows`) and ``capture`` receives the two buffers holding them, ``term`` and
        ``loss_weights``.  ``dw_acc``: the FP32 weight-gradient accumulator to use instead of a new one when its shape
        is the plan's (the zero-weight pass's arena, :func:`.cake_backend._arena_views`).

        Host-count path (``hidden`` None, :func:`forward_early_keys`): chunk 0's row gather and logits GEMM are bound
        and launched first, right behind the valid-row count, and the rest of the call -- the policy operands'
        gathers, the row vectors, the accumulators and the remaining launches -- is bound behind the queued GEMM
        instead of in front of it; the compacted labels are bound in front of the first early launch that reads them
        (:attr:`forward_early_labels`, the label-column arm) and behind the early launches otherwise.  The same
        kernels with the same operands on the same stream; the early launches neither read nor write what moved
        behind them, so every output is bitwise the in-order sequence's."""
        p, plan = self.plan.problem, self.plan
        T, H = (
            plan.rows,
            p.hidden,
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
                v0, v1 = plan.dw_acc_rows(0)
                dw_shape = (v1 - v0, H)
                t["dw_acc"] = (
                    dw_acc
                    if dw_acc is not None and tuple(dw_acc.shape) == dw_shape
                    else torch.empty(dw_shape, dtype=torch.float32, device=self.device)
                )
                if plan.dw_split_upper:
                    # the split accumulator's upper rows: an allocation of its own (:func:`_dw_cast_split`)
                    u0, u1 = plan.dw_acc_rows(_UpperRows(0))
                    t["dw_acc_hi"] = torch.empty(
                        (u1 - u0, H), dtype=torch.float32, device=self.device
                    )
            if plan.keep_row_grad:
                t["d_rows"] = torch.empty((T,), dtype=torch.float32, device=self.device)
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
                (t["lse"], t["d_rows"], t["labels"]) if plan.keep_row_grad else None,
                t.get(
                    "dw_acc_hi"
                ),  # the split accumulator's upper rows (:attr:`Plan.dw_split`)
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
        self._scratch(t, hidden)
        if plan.need_dx:
            t["dx_acc"] = torch.empty((T, H), dtype=torch.float32, device=self.device)
            if plan.dx_cast:
                t["dx_out"] = torch.empty(
                    (T, H), dtype=torch.bfloat16, device=self.device
                )
        if plan.need_dw:
            if plan.dw_acc_needed:
                v0, v1 = plan.dw_acc_rows(0)
                t["dw_acc"] = torch.empty(
                    (v1 - v0, H), dtype=torch.float32, device=self.device
                )
                if plan.dw_split_upper:
                    # the split accumulator's upper rows: an allocation of its own (:func:`_dw_cast_split`)
                    u0, u1 = plan.dw_acc_rows(_UpperRows(0))
                    t["dw_acc_hi"] = torch.empty(
                        (u1 - u0, H), dtype=torch.float32, device=self.device
                    )
            if (
                plan.dw_rows is None and plan.dw_split is None
            ):  # a ``dw_rows`` / ``dw_split`` plan leaves its accumulator(s) to the caller
                t["dw_out"] = torch.empty(
                    (V, H), dtype=torch.bfloat16, device=self.device
                )
        self._launch(self.backward_order, t, dw_stream)
        dx = (
            t.get("dx_out") if plan.dx_cast else t.get("dx_acc")
        )  # dx_cast=False: the caller finalizes the accumulator
        return dx, _logprob_dw_result(plan, t)


class _TemperatureBinding(_Binding):
    """A remembered binding of a call with a temperature (:attr:`Problem.temperature`, the temperature forms): each call
    passes its compact FP32 temperature rows (``temperature=``, the rows of the chunk loop like ``lse``), bound before
    the first launch (:meth:`_temperature`); every launch slots them (:func:`_slot_values_temperature`).  A binding
    without the temperature forms (:class:`_Binding`) takes no ``temperature``."""

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
            # fields of the base ``cake_backend.Plan`` (unseen when mypy skips the import)
            plan = replace(  # type: ignore[call-arg]
                plan,
                hidden_count=True,
                hidden_logits_group_m=hidden.group_m,
                hidden_stats=hidden.stats,
            )
            forward_order = _forward_keys_temperature(plan)
            backward_order = (
                _with_split_keys(plan, _recompute_keys_temperature(plan))
                if p.entry == "logprob"
                else ()
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

    def _temperature(
        self, t: dict[str, Any], temperature: Optional[torch.Tensor]
    ) -> None:
        """Check the call's compact temperature rows against the remembered plan and bind them."""
        p = self.plan.problem
        if (temperature is not None) != p.temperature:
            raise ValueError(
                "the remembered binding was planned for a call "
                + ("with" if p.temperature else "without")
                + " a temperature"
            )
        if temperature is not None:
            _check_output(temperature, "temperature", (self.plan.rows,), torch.float32)
            t["temperature"] = temperature

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
        dw_acc: Optional[torch.Tensor] = None,
    ):
        """One forward through the remembered launches; ``dw_stream`` is the call's dW side-stream mode
        (:func:`dw_stream_mode`), a launch placement outside the binding; ``temperature`` the call's compact FP32
        temperature rows (a binding planned with a temperature needs them, one without refuses them), bound before
        the first launch that reads them; ``dw_acc`` as in :meth:`_Binding.forward`.

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
        self._temperature(t, temperature)
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
                dw_shape = (V if plan.dw_rows is None else plan.dw_rows, H)
                t["dw_acc"] = (
                    dw_acc
                    if dw_acc is not None and tuple(dw_acc.shape) == dw_shape
                    else torch.empty(dw_shape, dtype=torch.float32, device=self.device)
                )
            if plan.keep_row_grad:
                t["d_rows"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        self._launch(self.forward_late if early else self.forward_order, t, dw_stream)
        if p.entry == "loss":
            return (
                t["loss"],
                t["logp"],
                t.get("dx_acc"),
                t.get("dw_acc"),
                _deferred_operands(plan, t),
                (t["lse"], t["d_rows"], t["labels"]) if plan.keep_row_grad else None,
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
        temperature: Optional[torch.Tensor] = None,
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
        self._temperature(t, temperature)
        self._compact(t, row_index)
        if plan.compact:
            t["d_in_full"] = dlogp
            t["d_in"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        else:
            t["d_in"] = dlogp
        self._scratch(t, hidden)
        if plan.need_dx:
            t["dx_acc"] = torch.empty((T, H), dtype=torch.float32, device=self.device)
            if plan.dx_cast:
                t["dx_out"] = torch.empty(
                    (T, H), dtype=torch.bfloat16, device=self.device
                )
        if plan.need_dw:
            if plan.dw_acc_needed:
                v0, v1 = plan.dw_acc_rows(0)
                t["dw_acc"] = torch.empty(
                    (v1 - v0, H), dtype=torch.float32, device=self.device
                )
                if plan.dw_split_upper:
                    # the split accumulator's upper rows: an allocation of its own (:func:`_dw_cast_split`)
                    u0, u1 = plan.dw_acc_rows(_UpperRows(0))
                    t["dw_acc_hi"] = torch.empty(
                        (u1 - u0, H), dtype=torch.float32, device=self.device
                    )
            if (
                plan.dw_rows is None and plan.dw_split is None
            ):  # a ``dw_rows`` / ``dw_split`` plan leaves its accumulator(s) to the caller
                t["dw_out"] = torch.empty(
                    (V, H), dtype=torch.bfloat16, device=self.device
                )
        self._launch(self.backward_order, t, dw_stream)
        dx = (
            t.get("dx_out") if plan.dx_cast else t.get("dx_acc")
        )  # dx_cast=False: the caller finalizes the accumulator
        return dx, _logprob_dw_result(plan, t)

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

        _run_split_keys(
            self.plan,
            self.device_index,
            keys,
            run_key,
            dw_stream,
            run_keys=_run_keys_temperature,
        )


class BindingCache:
    """Remembered input bindings of the eager entry points (least recently used, bounded).

    ``capacity`` (default :data:`BINDING_CACHE_DEFAULT_CAPACITY`, 64 bindings) bounds what is kept; ``enabled=False``
    bypasses the cache (every call validates and binds).  :meth:`configure` changes both at runtime.  The templated
    launches of the eager output-boundary casts (:func:`_eager_launch`) are kept alongside, at most
    :attr:`EAGER_CAPACITY` forms, and follow the same switch and :meth:`clear`.  Threads share it without a lock, as
    :class:`.cake_backend.BindingCache`'s (a step whose key or entry another thread's eviction removed is skipped)."""

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


# This module's bindings (their plans carry the opt-in forms), apart from :mod:`.cake_backend`'s cache; the same
# class (copied, so that it names this module's ``_Binding``) and default capacity, configured the same way.
BINDING_CACHE = BindingCache()


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


def _row_state(plan: Plan, t: dict[str, Any]) -> Optional[tuple]:
    """``(lse, d_rows, labels)`` of a ``keep_row_grad`` forward -- the ``[rows]`` row statistic, per-row gradient scale
    and prepared labels the budget forms' recompute reads (:func:`_dw_vocab_blocks`) -- else ``None``."""
    if not plan.keep_row_grad:
        return None
    return t["lse"], t["d_rows"], t["labels"]


class _BudgetResultState:
    """``memory_mode="budget"`` state of a :class:`ForwardResult`: the form :func:`memory_plan` picked (``None`` = the
    fast form).  Its recompute forms (``memory_plan.recompute``) also carry the chunk loop's ``[rows]`` row statistics
    ``(lse, d, labels)``, the plan the forward ran and its inputs ``(X, W)``: the backward recomputes the weight-gradient
    rows ``[dw_rows, V)``.  A plain base class: typed class defaults (``None``), not dataclass fields, so building the
    fast form's result costs what it did before; a budget forward sets them on its result."""

    memory_plan: Optional[MemoryPlan] = None
    row_state: Optional[tuple] = None
    plan: Optional[Plan] = None
    inputs: Optional[tuple] = None
    # a split accumulator (``"fused_ip"`` / ``"partial_ip"``, :attr:`Plan.dw_split`): its rows ``(a, end)`` -- the
    # lower part ``dw_acc`` the FP32 ``[a, H]`` rows ``[0, a)``, the upper part ``dw_acc_hi`` the FP32 ``[end - a, H]``
    # rows ``[a, end)`` (``None`` when ``end == a``); ``"partial_ip"`` recomputes the rows ``[end, V)``
    dw_split: Optional[tuple] = None
    dw_acc_hi: Optional[torch.Tensor] = None


@dataclass
class ForwardResult(_BudgetResultState, _main._TemperatureResultState):
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
    # ``memory_mode="budget"``: ``memory_plan`` / ``row_state`` / ``plan`` / ``inputs`` (:class:`_BudgetResultState`)

    def backward(self, grad=None, *, grad_weight_dtype=torch.bfloat16, backend=None):
        """``(dX, dW)`` of this forward through :func:`backward_loss` (``grad`` = the upstream scalar, ``None`` = 1);
        a recompute form of ``memory_mode="budget"`` finishes ``dW`` through the vocabulary-block recompute (BF16).
        ``backend`` (default: the forward's) overrides the backend of the finishing casts; the budget forms whose
        ``dW`` the cake kernels finish -- the split accumulator's cast (``"fused_ip"`` / ``"partial_ip"``) and the
        vocabulary-block recompute (``"partial_ip"`` / ``"partial"`` / ``"recompute"``) -- take the ``cake`` backend
        only (``ValueError`` for ``"reference"``), as their forward does."""
        if self.loss is None:
            raise ValueError(
                "ForwardResult.backward belongs to the loss entry (use backward_logprob)"
            )
        if (
            self.dw_split is not None
            or (self.memory_plan is not None and self.memory_plan.recompute)
        ) and (self.backend if backend is None else backend) != "cake":
            form = None if self.memory_plan is None else self.memory_plan.form
            raise ValueError(
                f"the weight gradient of the budget form {form!r} is finished by the cake kernels: the budget forms "
                f"{_SPLIT_FORMS + ('partial', 'recompute')} take backend='cake' only"
            )
        if self.dw_split is not None and self.dz_last is not None:
            # a split accumulator (``"fused_ip"`` / ``"partial_ip"``), read, never written (a repeatable backward)
            if grad_weight_dtype != torch.bfloat16:
                raise ValueError(
                    "memory_mode='budget' produces a BF16 weight gradient (grad_weight_dtype=torch.bfloat16)"
                )
            dx = (
                None
                if self.dx_acc is None
                else finalize_dx(
                    self.dx_acc,
                    grad,
                    self.row_index,
                    self.row_valid,
                    self.num_rows,
                    backend=self.backend if backend is None else backend,
                )
            )
            x_last = (
                self.x_last
                if self.x_last is not None
                else torch.index_select(self.x_src, 0, self.x_idx)
            )
            dw = _dw_cast_split(
                self.dz_last,
                x_last,
                [self.dw_acc, self.dw_acc_hi],
                grad,
                in_place=False,
                gemm_tuning=self.gemm_tuning,
            )
            if self.memory_plan is not None and self.memory_plan.recompute:
                # ``"partial_ip"``: the rows above the accumulated ones (:func:`_dw_vocab_blocks`, pinned last chunk)
                X, W = self.inputs
                lse_c, d_c, labels_c = self.row_state
                _dw_vocab_blocks(
                    X,
                    W,
                    self.row_index,
                    labels_c,
                    lse_c,
                    d_c,
                    self.plan,
                    grad,
                    dw,
                    v_start=int(self.dw_split[1]),
                    block_rows=int(self.memory_plan.block_rows),
                    gemm_tuning=self.gemm_tuning,
                    dz_last=self.dz_last,
                    x_last=x_last,
                )
            return dx, dw
        if self.memory_plan is not None and self.memory_plan.recompute:
            if grad_weight_dtype != torch.bfloat16:
                raise ValueError(
                    "memory_mode='budget' produces a BF16 weight gradient (grad_weight_dtype=torch.bfloat16)"
                )
            dx = (
                None
                if self.dx_acc is None
                else finalize_dx(
                    self.dx_acc,
                    grad,
                    self.row_index,
                    self.row_valid,
                    self.num_rows,
                    backend=self.backend if backend is None else backend,
                )
            )
            X, W = self.inputs
            dw = _dw_with_recompute(
                [] if self.dw_acc is None else [self.dw_acc],
                grad,
                X=X,
                W=W,
                row_index=self.row_index,
                row_state=self.row_state,
                plan=self.plan,
                dw_rows=self.memory_plan.dw_rows,
                block_rows=self.memory_plan.block_rows,
                gemm_tuning=self.gemm_tuning,
            )
            return dx, dw
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


def _hidden_count_program(
    problem: Problem,
    X: torch.Tensor,
    *,
    backend: str,
    compact: bool,
    stats: bool,
    hidden_count: bool,
) -> Optional[tuple[str, dict[str, Any]]]:
    """``(module_name, record)`` of a call that runs the hidden valid-row count (:func:`_hidden_count`'s rules: a
    compacted ``cake`` call with the switch on that :func:`hidden_count_eligible` admits), else ``None``.  Host checks
    only: nothing is allocated or queued."""
    if not (compact and backend == "cake" and hidden_count):
        return None
    rows, chunk = int(problem.num_rows), int(problem.chunk)
    if rows <= 0 or (rows + chunk - 1) // chunk < HIDDEN_COUNT_MIN_CHUNKS:
        return None  # the chunk rule of :func:`hidden_count_eligible`, ahead of the record lookup
    module_name, record = record_for(X.device)
    if not hidden_count_eligible(problem, X, record, module_name, stats=stats):
        return None
    return module_name, record


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
    :func:`valid_rows` path.  ``loss_weights``: the zero-weight split (:func:`.cake_backend.gradient_index`)."""
    program = _hidden_count_program(
        problem,
        X,
        backend=backend,
        compact=compact,
        stats=stats,
        hidden_count=hidden_count,
    )
    if program is None:
        return None
    module_name, record = program
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
    stats: bool,
    hidden_count: bool,
    temperature: TemperatureForm = None,
) -> Optional[_HiddenCount]:
    """:func:`_hidden_count` of a log-probability call with a temperature.  The hidden valid-row count prologue of a
    compacted ``cake`` call when the switch (``hidden_count``, resolved by
    the caller) is on and the call is eligible (:func:`hidden_count_eligible`); ``None`` = the shipped
    :func:`valid_rows` path.  ``temperature``: the call's resolved form (:func:`_resolve_temperature`; the problem
    records that it has one)."""
    if (temperature is not None) != problem.temperature:
        raise ValueError("the temperature form and the problem disagree")
    if not (compact and backend == "cake" and hidden_count):
        return None
    rows, chunk = int(problem.num_rows), int(problem.chunk)
    if rows <= 0 or (rows + chunk - 1) // chunk < HIDDEN_COUNT_MIN_CHUNKS:
        return None  # the chunk rule of :func:`hidden_count_eligible`, ahead of the record lookup
    module_name, record = record_for(X.device)
    eligible = (
        hidden_count_eligible_temperature
        if problem.temperature
        else hidden_count_eligible
    )
    if not eligible(problem, X, record, module_name, stats=stats):
        return None
    return _hidden_count_begin_temperature(
        problem,
        X,
        W,
        labels,
        record=record,
        module_name=module_name,
        stats=stats,
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
    memory_mode: Optional[str] = None,
    num_valid_rows=None,
    check_num_valid_rows: bool = False,
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

    ``memory_mode`` (default ``None``: the fast form; the autograd entries resolve theirs, :func:`memory_mode_default`,
    to :data:`MEMORY_AUTO` and pass it): ``"budget"`` (``cake`` backend, BF16 ``dW``) picks the weight-gradient form of
    :func:`memory_plan` once the call's row count is known (without the hidden valid-row count: its chunk-0 buffers
    come before the count; ``hidden_count`` is ignored) against the budget of the valid rows, also when uncompacted
    (``compact_rows=False``: the caller's ``num_valid_rows``, else one host count; :func:`_budget_valid_rows`) --
    ``fuse_dw_cast=False`` removes the fused form from the candidates -- and the result carries it (``memory_plan``);
    its recompute forms accumulate the vocabulary rows ``[0, dw_rows)`` only (``dw_acc`` is ``[dw_rows, H]``, ``None``
    for ``dw_rows = 0``) and keep the row statistics, the plan and the inputs for :meth:`ForwardResult.backward`,
    which recomputes the other rows (bitwise the same ``dW``).  :data:`MEMORY_AUTO` (the automatic form, on the
    ``cake`` backend with a CUDA ``X`` and a BF16 ``dW``; elsewhere the fast form, without raising) runs the fast form
    where it fits that budget with its own buffer lifetimes, else the budget form that fits (:func:`auto_memory_plan`;
    none fits: the fast form, ``memory_plan`` ``None``), on the valid rows of the count formed here or the caller's --
    an uncompacted call without the caller's count keeps the fast form, it never counts on the host.  It keeps the
    hidden valid-row count where the count's chunk-0 buffers fit the budget of any valid-row count
    (:func:`_hidden_count_fits`; checked on every pass of a step whose ``W`` requires a gradient, a checkpointed
    step's gradient-free first pass included) -- a call that misses the binding cache prepares its binding on views
    of those buffers (``_chunk_views`` of :func:`prepare_lm_head_det_loss`) --, plans a budget form
    with those chunk-sized buffers and drops the count's queued chunk-0 work only when that plan does not fit and the
    plan without the count does.

    ``num_valid_rows`` (a host integer, default ``None``): the number of labels ``>= 0``, given by the caller.  A
    compacted call then plans on it without reading a count back -- the valid-row index is formed on the device and the
    count is checked there by one asynchronous assertion (:func:`given_valid_rows`; a wrong count ends the CUDA context
    with a device-side assertion) -- so the forward does not wait for the device; ``check_num_valid_rows=True`` reads
    the count back instead and raises ``ValueError`` on a mismatch (one synchronization, for debugging).  An
    uncompacted call forms no index from the count (an uncompacted call of a budget form, explicit or automatic, sizes
    its budget on it); there
    only ``check_num_valid_rows=True`` checks it -- one readback of the labels' count, the same ``ValueError``
    (:func:`_check_count_on_host`) -- so with the flag every call checks its count once, whatever ``compact_rows``,
    ``memory_mode`` or ``need_dw``.  ``None``: the count is formed here (one synchronization per call;
    :func:`valid_rows`, the hidden valid-row count).

    Zero-weight policy rows (:func:`.cake_backend.forward_loss`; ``return_logp`` and the loss mode as there): left out
    of the gradient path when the count is formed here (not with ``num_valid_rows``); a call of a budget form plans its
    chunk loop on the gradient rows and its budget on every valid row (:func:`memory_plan`'s ``zero_rows``).  The
    automatic form first checks the fast call the memory rule picked (the split or the call without it, with the
    hidden count's chunk-0 buffers and the logp-only pass where they run) -- the rule's decision with the hidden count
    of :mod:`.cake_backend`'s default call, also when the check above dropped it, so a call that stays fast gives
    that call's bits; when it does not fit and a budget plan does, the call runs as a ``"budget"`` call (without the
    hidden count), whose split decision can differ."""
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
    # the opt-in keywords cost the default path (both ``None``) a few tests
    given = None if num_valid_rows is None else _given_count(num_valid_rows, T)
    # ``memory_mode=None`` of this eager entry: the fast form (the autograd entries resolve their default to
    # :data:`MEMORY_AUTO` and pass it)
    mode = None if memory_mode is None else _resolve_memory_mode(memory_mode)
    budget = auto = False
    if mode == "budget":
        _check_budget_call(backend, grad_weight_dtype)
        budget = bool(need_dw)
    elif mode == MEMORY_AUTO:
        # the automatic form: a budget form only where one can run (the generated program, a BF16 ``dW``), else the
        # fast form without raising
        auto = bool(
            need_dw
            and backend == "cake"
            and grad_weight_dtype == torch.bfloat16
            and X.is_cuda
        )
    # the hidden valid-row count of a default call whose step forms a BF16 weight gradient -- in this pass, or in a
    # later one of a checkpointed step (its first pass forms none, its ``W`` still requires one) -- is checked against
    # the step's budget (:func:`_auto_hidden_count`; the step's ``dX`` when ``X`` requires a gradient)
    count_checked = bool(
        mode == MEMORY_AUTO
        and backend == "cake"
        and grad_weight_dtype == torch.bfloat16
        and X.is_cuda
        and (need_dw or W.requires_grad)
    )
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
    zero_index = None
    # the hidden valid-row count the step-budget check dropped (the default call of :mod:`.cake_backend` runs with it)
    count_dropped = False
    if given is None:
        # tensors on another device: the plain path raises the placement error
        split = (
            zero_weight_split(problem, compact)
            and loss_weights is not None
            and loss_weights.device == labels.device == X.device
        )
        # ``"budget"``: no hidden valid-row count -- its chunk-0 buffers take ``min(chunk, T)`` rows before the count
        # is known, which can exceed the budget of a call with fewer valid rows than a chunk; the automatic form keeps
        # it where those buffers fit the budget of any valid-row count (:func:`_hidden_count_fits`; a cache miss
        # prepares its binding on views of them)
        count = False if budget else _resolve_hidden_count(hidden_count)
        if count and count_checked:
            count = bool(
                _auto_hidden_count(
                    count,
                    X,
                    W,
                    chunk_size,
                    need_dx=bool(need_dx or X.requires_grad),
                    entry="loss",
                )
            )
            count_dropped = not count
        hc = _hidden_count(
            problem,
            X,
            W,
            labels,
            backend=backend,
            compact=compact,
            stats=True,
            hidden_count=count,
            loss_weights=loss_weights if split else None,
        )
        if hc is not None:
            # the hidden valid-row count: chunk 0's gather and logits GEMM are queued; the count arrives through the
            # pinned cell
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
    else:
        # the caller's count: the plan needs no readback; the index is formed on the device behind the queued work
        hc = None
        row_index, row_valid = (
            given_valid_rows(labels, given, mask=fused_dx, check=check_num_valid_rows)
            if compact
            else (None, None)
        )
        if check_num_valid_rows and not compact:
            _check_count_on_host(labels, given)
    zero_rows = 0 if zero_index is None else int(zero_index.numel())
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    dx_cap: Optional[int] = None
    zero_order = zero_arena = False
    mplan: Optional[MemoryPlan] = None
    if zero_rows:
        # the memory rule, on the host from the two counts (:func:`.cake_backend.zero_weight_plan`): the split's chunk
        # loop runs at most the dX K slices per chunk of the call without the split, and the call splits only when the
        # split's predicted peak stays within that call's -- the fast form's byte model, or a budget call's
        # :func:`memory_plan` with the split and without it (the same budget, the valid rows) -- else it runs that call
        # (its valid-row index from the counts, no further synchronization; the hidden count's chunk-0 work is dropped)
        grad_rows = int(row_index.numel())
        valid_total = grad_rows + zero_rows
        exact_mode = zero_weight_loss_mode() == "exact"

        def zero_plan(with_count: bool) -> dict[str, Any]:
            return zero_weight_plan(
                problem,
                X.device,
                grad_rows=grad_rows,
                zero_rows=zero_rows,
                backend=backend,
                need_dx=need_dx,
                need_dw=need_dw,
                fuse_dw_cast=fuse,
                dx_finalize=fused_dx,
                hidden_count=with_count,
                return_logp=bool(return_logp),
                exact_loss=exact_mode,
                tuning=tuning,
            )

        zplan = zero_plan(hc is not None)
        dx_cap, zero_order, split_ok = (
            zplan["dx_slices_cap"],
            bool(zplan["zero_first"]),
            bool(zplan["split"]),
        )
        if (
            count_dropped
            and _hidden_count_program(
                problem,
                X,
                backend=backend,
                compact=compact,
                stats=True,
                hidden_count=True,
            )
            is not None
        ):
            # the split decision -- and with it the dX / dW bits -- of the default call of :mod:`.cake_backend`, which
            # runs with the hidden count dropped here: the rule's model with that count (the logp-only pass's order and
            # arena stay this call's, without it; ``dx_slices_cap`` does not depend on the count)
            split_ok = bool(zero_plan(True)["split"])

        def budget_of(rows: int, *, split_call: bool) -> MemoryPlan:
            return _budget_plan(
                "loss",
                X.device,
                num_rows=T,
                rows=rows,
                hidden=problem.hidden,
                vocab=problem.vocab,
                chunk=problem.chunk,
                compact=rows < T,
                need_dx=need_dx,
                fuse_dw_cast=fuse,
                buffer_rows=min(int(problem.chunk), rows),
                x_copy=problem.x_copy,
                dx_finalize=fused_dx,
                budget_rows=valid_total,
                tuning=tuning,
                zero_rows=zero_rows if (split_call and return_logp) else 0,
                zero_total=zero_rows if split_call else 0,
                exact_loss=split_call and exact_mode,
                dx_slices_cap=dx_cap if split_call else None,
            )

        plans: Optional[tuple[MemoryPlan, MemoryPlan]] = None
        if auto and grad_rows:
            # the automatic form: the fast call the rule above picked (the split, or the call without it) with its own
            # lifetimes -- the kept FP32 dX accumulator, the hidden count's chunk-0 buffers and where the logp-only
            # pass runs; a budget form only when that call does not fit and one of the budget call's two plans does
            # (else the fast call, the hidden count kept)
            C = int(problem.chunk)
            if _auto_plans(grad_rows if split_ok else valid_total, C):
                shape: dict[str, Any] = dict(
                    num_rows=T,
                    hidden=problem.hidden,
                    vocab=problem.vocab,
                    chunk=C,
                    need_dx=need_dx,
                    fuse_dw_cast=fuse,
                    x_copy=problem.x_copy,
                    dx_finalize=fused_dx,
                    budget_rows=valid_total,
                    tuning=tuning,
                )
                if split_ok:
                    fast_fits = _fast_fits(
                        "loss",
                        X.device,
                        rows=grad_rows,
                        compact=True,
                        buffer_rows=C if hc is not None else min(C, grad_rows),
                        zero_rows=zero_rows if return_logp else 0,
                        zero_total=zero_rows,
                        exact_loss=exact_mode,
                        dx_slices_cap=dx_cap,
                        hidden_count=hc is not None,
                        zero_pass=(
                            ("arena" if zplan["pass_arena"] else "first")
                            if zero_order
                            else "after"
                        ),
                        **shape,
                    )
                else:
                    fast_fits = _fast_fits(
                        "loss",
                        X.device,
                        rows=valid_total,
                        compact=valid_total < T,
                        buffer_rows=min(C, valid_total),
                        **shape,
                    )
                if not fast_fits:
                    split_plan = budget_of(grad_rows, split_call=True)
                    plain_plan = budget_of(valid_total, split_call=False)
                    if split_plan.fits or plain_plan.fits:
                        budget, plans = True, (split_plan, plain_plan)
        # a budget call's accumulator rows follow its plan: no arena, and the pass ahead of the loop as its plan models
        zero_arena = bool(zplan["pass_arena"]) and not budget
        if budget:
            zero_order = bool(return_logp)
        if budget and grad_rows:
            # (every valid row of zero weight: no chunk loop, the fast form's byte model above decides)
            mplan, plain_mplan = (
                plans
                if plans is not None
                else (
                    budget_of(grad_rows, split_call=True),
                    budget_of(valid_total, split_call=False),
                )
            )
            split_ok = mplan.predicted_bytes <= plain_mplan.predicted_bytes
            if not split_ok:
                mplan = plain_mplan
            # the automatic form's budget call runs without the hidden count, as its plans model it (its row indices
            # were formed above; the count's chunk-0 work is dropped)
            hc = None
        if not split_ok:
            hc = None
            dx_cap, zero_order, zero_arena = None, False, False
            row_index, row_valid = _valid_rows_of_count(
                labels, valid_total, mask=fused_dx
            )
            zero_index, zero_rows = None, 0
    elif auto:
        # the automatic form of a call without zero-weight rows (:func:`_auto_loss_plan`) on the valid rows of the
        # count formed above or the caller's; an uncompacted call without the caller's count keeps the fast form (it
        # never counts on the host).  A budget form keeps the hidden count when its plan of chunk-sized buffers fits,
        # else it drops the count's queued chunk-0 work (its row index was formed above)
        rows = T if row_index is None else int(row_index.numel())
        if rows and (row_index is not None or given is not None or compact):
            mplan, keep_hc = _auto_loss_plan(
                problem,
                X.device,
                rows=rows,
                valid=_budget_valid_rows(labels, row_index, given, compact),
                hidden_count=hc is not None,
                compact=row_index is not None,
                need_dx=need_dx,
                fuse_dw_cast=fuse,
                dx_finalize=fused_dx,
                tuning=tuning,
            )
            budget = mplan is not None
            if budget and not keep_hc:
                hc = None
    exact = bool(zero_rows) and zero_weight_loss_mode() == "exact"
    # the zero-weight rows' logp-only pass: ahead of the chunk loop unless the hidden valid-row count holds chunk 0's
    # buffers and the loop leaves none alive until the backward, or no chunk loop follows (every valid row has zero
    # weight) -- then after it / in place of it, with ``hc`` released (:func:`.cake_backend.forward_loss`).  A budget
    # call runs without the hidden count: always ahead of the loop, as :func:`memory_plan` models it.
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
    options: Optional[dict[str, Any]] = None
    if budget and not no_loop:
        # ``memory_mode="budget"``: the form :func:`memory_plan` picks for this call; a recompute form accumulates the
        # weight-gradient rows ``[0, dw_rows)`` in the chunk loop (none: the backward recomputes every row) and keeps
        # every chunk's per-row gradient scale (``options``: the plan's ``dw_rows`` / ``keep_row_grad``)
        rows = T if valid_count is None else valid_count
        mplan = (
            mplan
            if mplan is not None
            else _budget_plan(
                "loss",
                X.device,
                num_rows=T,
                rows=rows,
                hidden=problem.hidden,
                vocab=problem.vocab,
                chunk=problem.chunk,
                compact=row_index is not None,
                need_dx=need_dx,
                fuse_dw_cast=fuse,
                buffer_rows=min(int(problem.chunk), rows),
                x_copy=problem.x_copy,
                dx_finalize=fused_dx,
                budget_rows=_budget_valid_rows(labels, row_index, given, compact)
                + zero_rows,
                tuning=tuning,
                zero_rows=zero_rows if return_logp else 0,
                zero_total=zero_rows,
                exact_loss=exact,
                dx_slices_cap=dx_cap,
            )
        )
        fuse = mplan.form == "fused"
        split_form = mplan.form in ("fused_ip", "partial_ip")
        if (
            split_form
            and need_dw
            and -(-(T if valid_count is None else valid_count) // int(problem.chunk))
            > 1
        ):
            # the split FP32 accumulator (:attr:`Plan.dw_split`): the backward writes ``dW`` into its lower part;
            # ``"partial_ip"`` accumulates the rows below its ``dw_rows`` only and keeps the per-row gradient scale
            # for the recompute of the others
            fuse = True
            options = dict(
                dw_split=dw_split_rows(
                    problem.vocab, Geometry.from_record(record_for(X.device)[1])
                )
            )
            if mplan.form == "partial_ip":
                options.update(dw_split_end=int(mplan.dw_rows), keep_row_grad=True)
        elif split_form:
            # a one-chunk call has no accumulator (nor does a call without a weight gradient): the fused form, whose
            # backward recomputes nothing
            fuse = True
            mplan = replace(
                mplan, form="fused", dw_rows=int(problem.vocab), block_rows=None
            )
        elif mplan.recompute:
            options = dict(dw_rows=mplan.dw_rows or None, keep_row_grad=True)
            if not mplan.dw_rows:
                common["need_dw"] = False
    # the exact loss of a zero-weight split: the chunk loop keeps its loss terms and hands their two buffers over
    terms: Optional[dict] = {} if exact else None
    cache = BINDING_CACHE
    key = None
    binding = None
    if backend == "cake" and cache.enabled and not no_loop:
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
        if options is not None and "dw_split" in options:
            # the split accumulator's row and end (``"partial_ip"``: with the saved row scale)
            key += (
                "dw_split",
                int(options["dw_split"]),
                options.get("dw_split_end"),
                bool(options.get("keep_row_grad")),
            )
        elif options is not None:
            key += (options["dw_rows"], True)  # a recompute form's plan options
        if dx_cap is not None:
            key += ("dx_slices_cap", int(dx_cap))  # the split's capped dX slice counts
        binding = cache.lookup(key)
    # the pass arena below (the loop's FP32 weight-gradient accumulator, allocated ahead of the logp-only pass)
    with_arena = bool(zero_arena and zero_first and backend == "cake")
    if binding is None and with_arena and not no_loop:
        # a call that misses the cache prepares its binding (below) before the arena exists, so the runner's workspace
        # and the arena never coexist
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
            _template_only=backend == "cake",
            dx_slices_cap=dx_cap,
            _chunk_views=hc if backend == "cake" else None,
            **common,
            **(options or {}),
        )
        binding = _Binding.from_runner(runner, hc)
        del runner
        if key is not None:
            cache.remember(key, binding)
    # ahead of a loop that keeps the FP32 weight-gradient accumulator: the pass's chunk buffers in it, allocated first
    # (:func:`.cake_backend.forward_loss`)
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
    if binding is None:
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
            _template_only=backend == "cake",
            dx_slices_cap=dx_cap,
            _chunk_views=hc if backend == "cake" else None,
            **common,
            **(options or {}),
        )
        if backend == "cake":
            # The first call launches through the templated binding too: its per-call temporaries come from the caching
            # allocator, so the deferred ``dz`` rows keep the chunk buffer alive and nothing else -- a view into the
            # runner's single workspace allocation would pin the whole workspace until the backward.  The runner (and
            # its workspace) is dropped here; only the binding survives.  The runner is template-only: its buffers share
            # one allocation of the largest, so a call that misses the cache (a new valid-row count, every new batch)
            # peaks close to one that hits it rather than at the sum of the runner's buffers.
            binding = _Binding.from_runner(runner, hc)
            del runner
            if key is not None:
                cache.remember(key, binding)
        else:
            if terms is not None:
                _keep_runner_terms(runner)
            loss, logp = runner.forward()
            dx_acc, dw_acc = runner.dx_acc, runner.dw_acc
            dw_acc_hi = runner.tensors.get("dw_acc_hi")
            deferred = _deferred_operands(runner.plan, runner.tensors)
            row_state = _row_state(runner.plan, runner.tensors)
            memory, plan = runner.memory, runner.plan
            if terms is not None:
                terms.update(
                    term=runner.tensors["term"],
                    loss_weights=runner.tensors["loss_weights"],
                )
            del runner
    if binding is not None:
        loss, logp, dx_acc, dw_acc, deferred, row_state, dw_acc_hi = binding.forward(
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
        memory, plan = binding.memory, binding.plan
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
    result = ForwardResult(
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
    )
    if mplan is not None:
        _budget_result(result, mplan, row_state, plan, X, W, dw_acc_hi)
    return _with_zero_weight_rows(
        result,
        zero_index,
        zero_logp,
        chunk0_alive=chunk0_alive,
        exact=exact,
        deferred_alive=zero_pass and not zero_first and fuse,
    )


def _keep_runner_terms(runner: LmHeadDetLossRunner) -> None:
    """A prepared (``reference`` backend) runner of a zero-weight split's chunk loop that keeps its loss terms
    (:func:`.cake_backend._term_rows`): its ``row_finalize`` / ``loss_reduce`` host values re-formed with
    ``keep_terms``."""
    t = runner.tensors
    t["keep_terms"] = True
    for key in runner.forward_order:
        if key[0] in ("row_finalize", "loss_reduce"):
            runner.values[key] = stage_values(key[0], t, runner.plan, key[1])


def _budget_result(
    result: ForwardResult,
    mplan: MemoryPlan,
    row_state,
    plan: Plan,
    X,
    W,
    dw_acc_hi: Optional[torch.Tensor] = None,
) -> None:
    """Record a ``memory_mode="budget"`` forward's form on its result; a recompute form also keeps the row statistics,
    plan and inputs the backward's vocabulary-block recompute reads, a split accumulator (``"fused_ip"`` /
    ``"partial_ip"``, :attr:`Plan.dw_split`) its rows ``(dw_split, end)`` and upper part ``dw_acc_hi`` (``dw_acc``
    holds the lower rows; ``None`` for an empty upper part)."""
    result.memory_plan = mplan
    if mplan.recompute:
        result.row_state, result.plan, result.inputs = row_state, plan, (X, W)
    if plan is not None and plan.dw_split is not None:
        result.dw_split = (
            int(plan.dw_split),
            int(plan.dw_acc_rows(_UpperRows(0))[1]),
        )
        result.dw_acc_hi = dw_acc_hi


def _check_budget_call(backend: str, grad_weight_dtype) -> None:
    """``memory_mode="budget"`` runs the generated program (its recompute launches kernels) and writes a BF16 ``dW``."""
    if backend != "cake":
        raise ValueError(
            "memory_mode='budget' runs on the cake backend (the reference backend runs the fast form only)"
        )
    if grad_weight_dtype != torch.bfloat16:
        raise ValueError(
            "memory_mode='budget' produces a BF16 weight gradient (grad_weight_dtype=torch.bfloat16)"
        )


@functools.lru_cache(maxsize=1024)
def _dx_ws_slabs(
    rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    num_sms: int,
    dx_max: int,
    geometry: Geometry,
    resident: int,
    explicit_k: Optional[int],
    cap: Optional[int] = None,
) -> int:
    """The K-slice slabs of a plan's dX GEMM (:attr:`Plan.dx_ws_slabs`) without building the plan: the slice rule of
    :func:`make_plan` (an explicit ``k_slices`` knob wins; ``cap``: its ``dx_slices_cap``)."""
    if explicit_k is not None:
        return int(explicit_k) - 1
    if int(dx_max) <= 1:
        return 0
    return (
        max(
            capped_k_slices(
                recommended_k_slices(
                    rows_c, hidden, num_sms, dx_max, geometry, resident, vocab=vocab
                ),
                cap,
                vocab,
                geometry,
            )
            for _, rows_c in plan_chunks(rows, chunk)
        )
        - 1
    )


def _plan_inputs(
    device: torch.device,
    *,
    rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    need_dx: bool,
    tuning: dict[str, dict[str, Any]],
    dx_slices_cap: Optional[int] = None,
    temperature: bool = False,
) -> tuple[Geometry, int, bool]:
    """``(geometry, dx_ws_slabs, recompute)`` of :func:`memory_plan` for one call on ``device``'s registered program:
    its geometry, the dX GEMM's K-slice slabs of a chunk loop over ``rows`` rows (``dx_slices_cap``: the plan's,
    :func:`make_plan`) and whether the program registers the recompute's ``dz`` arm (``temperature``: its temperature
    form, :data:`LOGPROB_DZ_TAU_STAGE`, among the stages with a temperature)."""
    module_name, record = record_for(device)
    geometry = Geometry.from_record(record)
    stages = (
        registered_stages_with_temperature(module_name)
        if temperature
        else registered_stages(module_name)
    )
    slabs = 0
    if need_dx:
        num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        dx_tuning = tuning.get("dx", {})
        explicit_k = (
            (1 if dx_tuning["k_slices"] is None else int(dx_tuning["k_slices"]))
            if "k_slices" in dx_tuning
            else None
        )
        slabs = _dx_ws_slabs(
            int(rows),
            int(hidden),
            int(vocab),
            int(chunk),
            num_sms,
            dx_max_slices(stages),
            geometry,
            cluster_resident(device, geometry.dx_cluster_ctas, num_sms),
            explicit_k,
            None if dx_slices_cap is None else int(dx_slices_cap),
        )
    dz = LOGPROB_DZ_TAU_STAGE if temperature else LOGPROB_DZ_STAGE
    return geometry, slabs, dz in stages


def _budget_plan(
    entry: str,
    device: torch.device,
    *,
    num_rows: int,
    rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    compact: bool,
    need_dx: bool,
    fuse_dw_cast: bool,
    buffer_rows: int,
    x_copy: bool,
    dx_finalize: bool,
    budget_rows: int,
    tuning: dict[str, dict[str, Any]],
    zero_rows: int = 0,
    zero_total: int = 0,
    exact_loss: bool = False,
    dx_slices_cap: Optional[int] = None,
    hidden_count: bool = False,
) -> MemoryPlan:
    """:func:`memory_plan` of one call of a budget form on ``device``'s registered program (:func:`_plan_inputs`: its
    geometry, the dX GEMM's K-slice slabs of this call and whether the program registers the recompute's ``dz`` arm).
    ``rows``: the rows the chunk loop runs over; ``budget_rows``: the valid rows ``T_v`` of the budget
    (:func:`_budget_valid_rows`); ``zero_rows`` / ``zero_total`` / ``exact_loss`` / ``hidden_count`` (a default call
    that keeps the hidden valid-row count): as :func:`memory_plan`'s; ``dx_slices_cap``: the plan's
    (:func:`make_plan`)."""
    geometry, slabs, recompute = _plan_inputs(
        device,
        rows=rows,
        hidden=hidden,
        vocab=vocab,
        chunk=chunk,
        need_dx=need_dx,
        tuning=tuning,
        dx_slices_cap=dx_slices_cap,
    )
    return memory_plan(
        entry,
        num_rows=num_rows,
        rows=rows,
        hidden=hidden,
        vocab=vocab,
        chunk=chunk,
        need_dx=need_dx,
        compact=compact,
        fuse_dw_cast=fuse_dw_cast,
        recompute=recompute,
        dx_ws_slabs=slabs,
        buffer_rows=buffer_rows,
        x_copy=x_copy,
        dx_finalize=dx_finalize,
        budget_rows=budget_rows,
        zero_rows=zero_rows,
        zero_total=zero_total,
        exact_loss=exact_loss,
        geometry=geometry,
        hidden_count=hidden_count,
    )


def _budget_plan_temperature(
    entry: str,
    device: torch.device,
    *,
    num_rows: int,
    rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    compact: bool,
    need_dx: bool,
    fuse_dw_cast: bool,
    buffer_rows: int,
    x_copy: bool,
    dx_finalize: bool,
    budget_rows: int,
    tuning: dict[str, dict[str, Any]],
) -> MemoryPlan:
    """:func:`_budget_plan` of a call with a temperature: the recompute forms need the temperature form of the
    recompute's ``dz`` arm (:data:`LOGPROB_DZ_TAU_STAGE`) among the stages the program registers."""
    geometry, slabs, recompute = _plan_inputs(
        device,
        rows=rows,
        hidden=hidden,
        vocab=vocab,
        chunk=chunk,
        need_dx=need_dx,
        tuning=tuning,
        temperature=True,
    )
    return memory_plan(
        entry,
        num_rows=num_rows,
        rows=rows,
        hidden=hidden,
        vocab=vocab,
        chunk=chunk,
        need_dx=need_dx,
        compact=compact,
        fuse_dw_cast=fuse_dw_cast,
        recompute=recompute,
        dx_ws_slabs=slabs,
        buffer_rows=buffer_rows,
        x_copy=x_copy,
        dx_finalize=dx_finalize,
        budget_rows=budget_rows,
        geometry=geometry,
    )


def _fast_fits(
    entry: str,
    device: torch.device,
    *,
    temperature: bool = False,
    hidden_count: bool = False,
    zero_pass: str = "first",
    tuning: dict[str, dict[str, Any]],
    dx_slices_cap: Optional[int] = None,
    **kw: Any,
) -> bool:
    """:func:`fast_form_fits` of one default call on ``device``'s registered program (:func:`_plan_inputs`); ``kw``:
    :func:`_budget_plan`'s keywords, ``hidden_count`` / ``zero_pass``: :func:`memory_plan`'s.  It plans through
    :func:`memory_plan` directly, not through :func:`_budget_plan`."""
    geometry, slabs, _ = _plan_inputs(
        device,
        rows=kw["rows"],
        hidden=kw["hidden"],
        vocab=kw["vocab"],
        chunk=kw["chunk"],
        need_dx=kw["need_dx"],
        tuning=tuning,
        dx_slices_cap=dx_slices_cap,
        temperature=temperature,
    )
    return fast_form_fits(
        entry,
        dx_ws_slabs=slabs,
        geometry=geometry,
        hidden_count=hidden_count,
        zero_pass=zero_pass,
        **kw,
    )


def _auto_plan(
    entry: str, device: torch.device, *, temperature: bool = False, **kw: Any
) -> Optional[MemoryPlan]:
    """:func:`auto_memory_plan` of one default call on ``device``'s registered program without zero-weight rows:
    ``None`` (the fast form) when the call runs fewer than :data:`_AUTO_MIN_CHUNKS` chunks, when the fast form fits
    (:func:`_fast_fits`) or when no budget form fits; else the budget form (:func:`_budget_plan`, or
    :func:`_budget_plan_temperature` for a call with a temperature).  ``kw``: :func:`_budget_plan`'s keywords."""
    if not _auto_plans(kw["rows"], kw["chunk"]):
        return None
    if _fast_fits(entry, device, temperature=temperature, **kw):
        return None
    plan = (_budget_plan_temperature if temperature else _budget_plan)(
        entry, device, **kw
    )
    return plan if plan.fits else None


def _hidden_count_shape_fits(
    *,
    num_rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    need_dx: bool,
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> bool:
    """Whether the hidden valid-row count's work before the count is known stays within :func:`memory_plan`'s budget
    for a call of this shape whatever its valid rows ``T_v`` (pure host arithmetic; ``True`` for a call the count does
    not serve, fewer than :data:`HIDDEN_COUNT_MIN_CHUNKS` chunks).

    Before the count is known its chunk-0 buffers take ``chunk`` rows (logits, statistics, gather buffer) beside the
    call's ``O(T)`` vectors (``chunk0``).  That phase does not depend on ``T_v`` and the budget grows with it, so the
    budget of a call without a valid row bounds it for every count: ``chunk0 <= (1 - margin) * (2 * V * H + 2 * T *
    H)`` (the last term without ``dX``).  After the count, a call that misses the binding cache -- a new valid-row
    count, every new batch -- prepares its binding on views of those buffers (``_chunk_views`` of
    :func:`prepare_lm_head_det_loss`, of :func:`.cake_backend.prepare_lm_head_det_loss` for the log-probability
    forward), so no second chunk buffer exists beside them: that phase is :func:`memory_plan`'s ``miss`` phase, never
    above its forward phase."""
    T, H, V, C = int(num_rows), int(hidden), int(vocab), int(chunk)
    if T <= 0 or C <= 0 or -(-T // C) < HIDDEN_COUNT_MIN_CHUNKS:
        return True
    tile = int(geometry.stats_tile)
    live = _allocated_bytes
    chunk0 = (
        live(2 * C * V)
        + live(8 * C * -(-V // tile))
        + live(2 * C * H)
        + live(4 * T, 4)
        + live(4 * T, 8)
    )
    head = 2 * V * H + (2 * T * H if need_dx else 0)
    budget = head - int(math.ceil(head * float(MEMORY_BUDGET_MARGIN)))
    return chunk0 <= budget


def _hidden_count_fits(
    device: torch.device,
    *,
    num_rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    need_dx: bool,
    entry: str = "loss",
) -> bool:
    """:func:`_hidden_count_shape_fits` of a default call of ``entry`` on ``device``'s registered program (its
    geometry).  A default call whose count fails it runs without the hidden count (the shipped :func:`valid_rows`
    path, bitwise the same outputs), so the count's chunk-0 work never exceeds the budget of a call that a budget form
    fits.  ``entry``: the loss or log-probability entry (the same rule: the work before the count is the same)."""
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}")
    T, C = int(num_rows), int(chunk)
    if T <= 0 or C <= 0 or -(-T // C) < HIDDEN_COUNT_MIN_CHUNKS:
        return True
    _, record = record_for(device)
    return _hidden_count_shape_fits(
        num_rows=T,
        hidden=hidden,
        vocab=vocab,
        chunk=C,
        need_dx=need_dx,
        geometry=Geometry.from_record(record),
    )


def _auto_reaches(backend: str, X: Any, W: Any) -> bool:
    """Whether a default call (:data:`MEMORY_AUTO`) belongs to a step that forms a BF16 weight gradient on the
    ``cake`` backend: ``X`` on a CUDA device and a BF16 ``W`` that requires a gradient, whatever the gradient mode.
    With gradient mode on the call can switch to a budget form; with it off (e.g. a reentrant checkpoint's first pass)
    it runs the fast form, its hidden valid-row count -- part of the step's peak -- checked as with it on
    (:func:`_auto_hidden_count`).  Every other default call runs :mod:`.cake_backend`'s entry unchanged."""
    return (
        backend == "cake"
        and isinstance(X, torch.Tensor)
        and X.is_cuda
        and isinstance(W, torch.Tensor)
        and W.dtype == torch.bfloat16
        and W.requires_grad
    )


def _auto_hidden_count(
    hidden_count: Optional[bool],
    X: torch.Tensor,
    W: torch.Tensor,
    chunk_size: Any,
    *,
    need_dx: bool,
    entry: str = "loss",
) -> Optional[bool]:
    """``hidden_count`` of a default call whose step forms a BF16 weight gradient (:func:`_auto_reaches`), every pass
    of a checkpointed step included: ``False`` when the hidden valid-row count's chunk-0 work can exceed the step's
    budget (:func:`_hidden_count_fits` of ``entry``), else the caller's value."""
    if X.ndim != 2 or W.ndim != 2 or isinstance(chunk_size, bool):
        return hidden_count  # the validation raises
    try:
        chunk = int(chunk_size)
    except (TypeError, ValueError):
        return hidden_count
    if chunk < 1 or chunk != chunk_size:
        return hidden_count
    fits = _hidden_count_fits(
        X.device,
        num_rows=int(X.shape[0]),
        hidden=int(X.shape[1]),
        vocab=int(W.shape[0]),
        chunk=chunk,
        need_dx=need_dx,
        entry=entry,
    )
    return hidden_count if fits else False


def _auto_loss_plan(
    problem: Problem,
    device: torch.device,
    *,
    rows: int,
    valid: int,
    hidden_count: bool,
    compact: bool,
    need_dx: bool,
    fuse_dw_cast: bool,
    dx_finalize: bool,
    tuning: dict[str, dict[str, Any]],
) -> tuple[Optional[MemoryPlan], bool]:
    """The automatic form of a default loss call without zero-weight rows: ``(plan, keep_hidden_count)``.  ``plan``:
    ``None`` = the fast form.  ``rows``: the rows of its chunk loop, ``valid``: its valid rows ``T_v``,
    ``hidden_count``: the hidden valid-row count is alive (its chunk-0 buffers take ``chunk`` rows, its gather buffer
    exists also when every row is valid, and a call that misses the binding cache prepares its binding beside them:
    :func:`memory_plan`'s ``hidden_count``).  With the count, the plan of ``chunk``-row buffers keeps it when it fits;
    else the plan of ``min(chunk, rows)``-row buffers -- an explicit budget call's -- drops it (only once that plan is
    known to fit: a fast fallback keeps the count's queued chunk-0 work)."""
    C = int(problem.chunk)
    common: dict[str, Any] = dict(
        num_rows=int(problem.num_rows),
        rows=int(rows),
        hidden=int(problem.hidden),
        vocab=int(problem.vocab),
        chunk=C,
        need_dx=need_dx,
        fuse_dw_cast=fuse_dw_cast,
        x_copy=bool(problem.x_copy),
        dx_finalize=dx_finalize,
        budget_rows=int(valid),
        tuning=tuning,
    )
    if not _auto_plans(rows, C):
        return None, True
    shared = min(C, int(rows))
    if hidden_count:
        if _fast_fits(
            "loss", device, compact=True, buffer_rows=C, hidden_count=True, **common
        ):
            return None, True
        kept = _budget_plan(
            "loss", device, compact=True, buffer_rows=C, hidden_count=True, **common
        )
        if kept.fits:
            return kept, True
        if shared == C and compact:
            # the same plan without the count's phases: its cache-miss phase (the template on views of the count's
            # buffers) never exceeds its forward phase, so that plan does not fit either -- none fits
            return None, True
    elif _fast_fits("loss", device, compact=compact, buffer_rows=shared, **common):
        return None, True
    plan = _budget_plan("loss", device, compact=compact, buffer_rows=shared, **common)
    return (plan, False) if plan.fits else (None, True)


def _logprob_auto(mode: Optional[str], backend: str, X: Any, W: Any) -> bool:
    """Whether a log-probability call of ``mode`` runs the automatic form (:data:`MEMORY_AUTO`): its backward picks
    the fast form or a budget form.  The ``cake`` backend with a CUDA ``X`` and a BF16 ``W`` only; elsewhere the fast
    form, without raising."""
    return bool(
        mode == MEMORY_AUTO
        and backend == "cake"
        and isinstance(X, torch.Tensor)
        and X.is_cuda
        and isinstance(W, torch.Tensor)
        and W.dtype == torch.bfloat16
    )


def _auto_logprob_plan(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    row_index: Optional[torch.Tensor],
    *,
    compact: bool,
    given: Optional[int],
    need_dx: bool,
    fuse_dw_cast: bool,
    dx_finalize: Optional[bool],
    chunk: int,
    tuning: dict[str, dict[str, Any]],
    backend: str,
    temperature: bool = False,
) -> Optional[MemoryPlan]:
    """The automatic form of a default log-probability backward (:func:`auto_memory_plan`, after its forward: the
    budget covers the backward): ``None`` = the fast form, else the budget form of :func:`_backward_logprob_budget`.
    ``row_index``: the forward's compact rows (``None``: every row, or an uncompacted call); ``compact``: the call's
    switch; ``given``: the caller's count (an uncompacted call without it keeps the fast form -- it never counts on the
    host).  ``temperature``: a call with a temperature (its recompute stage)."""
    T = int(labels.numel())
    if backend != "cake" or T == 0:
        return None
    if row_index is None and given is None and not compact:
        return None
    rows = T if row_index is None else int(row_index.numel())
    valid = _budget_valid_rows(labels, row_index, given, compact)
    if rows == 0 or valid == 0 or not _auto_plans(rows, chunk):
        return None
    problem = validate_lm_head_det_inputs(
        X, W, labels, chunk_size=chunk, entry="logprob"
    )
    return _auto_plan(
        "logprob",
        X.device,
        temperature=temperature,
        num_rows=T,
        rows=rows,
        hidden=int(problem.hidden),
        vocab=int(problem.vocab),
        chunk=int(problem.chunk),
        compact=row_index is not None,
        need_dx=need_dx,
        fuse_dw_cast=fuse_dw_cast,
        buffer_rows=min(int(problem.chunk), rows),
        x_copy=bool(problem.x_copy),
        dx_finalize=_resolve_dx_finalize(dx_finalize),
        budget_rows=valid,
        tuning=tuning,
    )


def _budget_valid_rows(
    labels: torch.Tensor,
    row_index: Optional[torch.Tensor],
    given: Optional[int],
    compact: bool,
) -> int:
    """The valid rows ``T_v`` of a ``memory_mode="budget"`` call's budget (:func:`memory_plan`'s ``budget_rows``): the
    compacted rows (``row_index``; every row when a compacted call's count found none ignored), else the caller's
    ``num_valid_rows`` (``given``; the forward checked it with ``check_num_valid_rows``), else the labels ``>= 0``
    counted on the host -- one synchronization, taken by an uncompacted call (``compact_rows=False``) without the
    caller's count: its chunk loop runs over every row, its budget counts the logits of the valid rows only."""
    if row_index is not None:
        return int(row_index.numel())
    if given is not None:
        return int(given)
    if compact:
        return int(labels.numel())
    return int((labels >= 0).sum())


def scale_cast(
    acc: torch.Tensor,
    grad: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    *,
    backend: str = "cake",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``cast(g * acc)`` into a new tensor of ``out_dtype`` (one rounding); ``acc`` is never written.

    The single fp32 -> output cast of the backward, for ``dW`` and ``dX`` alike
    (``g`` = the incoming scalar gradient, ``None`` = 1).  ``out`` (a contiguous tensor of ``acc``'s element count
    and ``out_dtype``, e.g. a row block of a larger ``dW``) receives the cast instead of a new tensor and is returned.
    """
    if out_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("the cast produces bfloat16 or float32")
    if not acc.is_contiguous() or acc.dtype != torch.float32:
        raise ValueError("the accumulator must be a contiguous FP32 tensor")
    if out is None:
        out = torch.empty(acc.shape, dtype=out_dtype, device=acc.device)
    elif (
        out.dtype != out_dtype
        or out.numel() != acc.numel()
        or not out.is_contiguous()
        or out.device != acc.device
    ):
        raise ValueError(
            f"out must be a contiguous {out_dtype} tensor of {acc.numel()} elements on the accumulator's device"
        )
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
) -> torch.Tensor:
    """The dX output boundary: ``bf16(g * acc)`` restored to the caller's ``[num_rows, H]`` rows.  Every row valid
    (``row_index`` None): :func:`scale_cast`, already one pass over every row.  Compacted rows with the forward's
    valid-row mask (``row_valid``: the fused dX finalize): the inclusive int32 scan -- one ``cumsum``, issued here
    behind the backward's queued GEMMs -- and one :func:`scale_cast_scatter` launch.  Compacted rows without the mask
    (the forward ran with ``dx_finalize=False``): the flat cast, then :func:`scatter_rows`
    (zero fill + ``index_copy_``).  All three produce the same bits."""
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
    form = _DwCastForm.of(
        device,
        int(rows),
        H,
        fp32=out_dtype == torch.float32,
        multi=dw_acc is not None,
        gemm_tuning=gemm_tuning,
    )
    form.launch(dz_last, x_rows, dw_acc, out, g)
    return out


@dataclass(frozen=True)
class _DwCastForm:
    """The fused scale + cast GEMM of a deferred last chunk (:func:`dw_cast`, :func:`_dw_cast_split`) on one device:
    the record, its stage and launch scalars -- the raster / block rules of the record's architecture on the chunk's
    rows, or the explicit knobs (the block rule and knob apply to the fp32 cast only; the bf16 cast keeps the 1-D
    raster; the TMA store rule to a multi-chunk bf16 cast without an explicit ``epi_store``)."""

    module_name: str
    record: Any
    geometry: Geometry
    stage: str
    group_m: Optional[int]
    group_n: Optional[int]
    device: torch.device
    index: int

    @classmethod
    def of(
        cls,
        device: torch.device,
        rows: int,
        hidden: int,
        *,
        fp32: bool,
        multi: bool,
        gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    ) -> "_DwCastForm":
        module_name, record = record_for(device)
        record_abi(record)
        dw_tuning = _resolve_gemm_tuning(gemm_tuning).get("dw", {})
        raster = raster_variant(int(hidden), int(rows), record["arch"])
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
                else dw_block_variant(int(rows), int(hidden), record["arch"])
            )
            group_n = None if group_n in (None, 0) else int(group_n)
        stage = stage_variant(
            "gemm_dw_cast_f32" if fp32 else "gemm_dw_cast_bf16",
            blocked=group_n is not None,
            epi_store="tma"
            if (
                not fp32
                and multi
                and "epi_store" not in dw_tuning
                and dw_cast_tma_rule(2, int(rows), record["arch"])
            )
            else None,
        )
        if stage not in registered_stages(module_name):
            raise NotImplementedError(
                f"the registered program {module_name!r} lacks the stage {stage!r} (fuse_dw_cast)"
            )
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return cls(
            module_name,
            record,
            Geometry.from_record(record),
            stage,
            group_m,
            group_n,
            device,
            int(index),
        )

    def launch(
        self,
        dz: torch.Tensor,
        x_rows: torch.Tensor,
        acc: Optional[torch.Tensor],
        out: torch.Tensor,
        g: torch.Tensor,
    ) -> None:
        """One launch over the vocabulary rows of ``dz`` (BF16 ``[rows, m]``, a column block of the chunk's ``dz``):
        ``out`` = ``cast(g * (acc + dz^T @ x_rows))`` (``[m, H]``; ``acc`` ``None``: ``cast(g * dz^T @ x_rows)``)."""
        rows, m = (int(n) for n in dz.shape)
        H = int(x_rows.shape[1])
        values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
        values.update({name: 0 for name in COMMON_SCALARS})
        values.update(
            A=dz,
            B=x_rows,
            C=out,
            STATS_OUT=g,
            WS=acc if acc is not None else _device_constants(self.index)["f32_dummy"],
            rows_c=rows,
            T=rows,
            H=H,
            V=m,
            loss_div=1.0,
            **gemm_scalars(
                self.stage,
                self.geometry,
                hidden=H,
                vocab=m,
                rows_c=rows,
                group_m=self.group_m,
                group_n=self.group_n,
                first_chunk=int(acc is None),
            ),
        )
        launch, arguments = _eager_launch(
            self.record,
            self.module_name,
            self.stage,
            values,
            self.device,
            self.geometry,
        )
        with _ffi_stream_context(self.index):
            if launch.prepare is not None:
                launch.prepare(*arguments)
            launch.entry(*arguments)


def _dw_cast_split(
    dz_last: torch.Tensor,
    x_rows: torch.Tensor,
    acc: list,
    grad: Optional[torch.Tensor],
    *,
    in_place: bool,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
) -> torch.Tensor:
    """The BF16 ``dW`` of a split accumulator (:attr:`Plan.dw_split`; the budget forms ``"fused_ip"`` /
    ``"partial_ip"``): :func:`dw_cast` of the deferred last chunk over ``acc`` = ``[lower, upper]`` -- the FP32
    ``[a, H]`` and ``[e - a, H]`` sums of the vocabulary rows ``[0, a)`` and ``[a, e)`` of the chunks before it
    (``upper`` ``None`` when ``e == a``; ``e = V`` for ``"fused_ip"``) -- as one fused GEMM per vocabulary row piece
    with the form, raster and store of the full launch (:class:`_DwCastForm`): each element sees the same FP32
    operations, so the rows ``[0, e)`` of ``dW`` are bitwise :func:`dw_cast`'s.  The rows ``[e, V)`` are left for the
    caller (``"partial_ip"``: :func:`_dw_vocab_blocks` with the deferred chunk's ``dz``).

    ``in_place`` (a backward that frees its graph): both tensors are taken out of ``acc`` and ``dW`` is written into
    the lower accumulator's storage (``4 a H >= 2 V H`` bytes; :func:`dw_split_rows`) -- the rows ``[0, a)`` in the
    pieces of :func:`_dw_split_pieces` in order, the first through a temporary copied into place after it, then the
    rows ``[a, e)`` from the upper accumulator, which is released after it -- and returned as a BF16 ``[V, H]`` tensor
    on that storage (autograd takes it as ``W.grad`` without a copy): the step holds no ``[V, H]`` output beside the
    accumulators, which this consumes.  Otherwise (a retained graph, :meth:`ForwardResult.backward`) the accumulators
    are read, never written, and ``dW`` is a new tensor (two launches)."""
    if len(acc) != 2:
        raise ValueError("the split accumulator's cast needs [lower, upper]")
    lower, upper = acc
    if in_place:
        acc.clear()  # the caller's list: dropped here, so this function holds the last references
    if (
        dz_last.dtype != torch.bfloat16
        or dz_last.ndim != 2
        or not dz_last.is_contiguous()
    ):
        raise ValueError("dz_last must be a contiguous BF16 [rows, V] tensor")
    rows, V = (int(n) for n in dz_last.shape)
    a = int(lower.shape[0])
    e = a if upper is None else a + int(upper.shape[0])
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
    if (
        lower.dtype != torch.float32
        or not lower.is_contiguous()
        or tuple(lower.shape) != (a, H)
        or (
            upper is not None
            and (
                upper.dtype != torch.float32
                or not upper.is_contiguous()
                or tuple(upper.shape) != (e - a, H)
            )
        )
        or not V <= 2 * a < 2 * V
        or not a <= e <= V
        or rows == 0
    ):
        raise ValueError(
            "the split accumulator is the contiguous FP32 [a, H] and [e - a, H] sums of a multi-chunk call, "
            "V / 2 <= a <= e <= V, a < V"
        )
    device = dz_last.device
    g = (
        _unit_scale(device)
        if grad is None
        else grad.detach().reshape(1).to(device=device, dtype=torch.float32)
    )
    form = _DwCastForm.of(
        device, rows, H, fp32=False, multi=True, gemm_tuning=gemm_tuning
    )
    if not in_place:
        dw = torch.empty((V, H), dtype=torch.bfloat16, device=device)
        form.launch(dz_last[:, :a], x_rows, lower, dw[:a], g)
        if upper is not None:
            form.launch(dz_last[:, a:e], x_rows, upper, dw[a:e], g)
        return dw
    storage = lower.untyped_storage()
    if lower.storage_offset() or storage.nbytes() < 2 * V * H:
        raise ValueError(
            "the lower accumulator must own the start of a storage of at least 2 V H bytes"
        )
    dw = torch.empty(0, dtype=torch.bfloat16, device=device).set_(
        storage, 0, (V, H), (H, 1)
    )
    for k, (m0, m1) in enumerate(_dw_split_pieces(a, dw_block_step(form.geometry))):
        if k == 0:
            # the first piece's BF16 rows would overwrite FP32 rows it still reads: through a temporary, copied into
            # place behind the launch (stream order)
            first = torch.empty((m1, H), dtype=torch.bfloat16, device=device)
            form.launch(dz_last[:, :m1], x_rows, lower[:m1], first, g)
            dw[:m1].copy_(first)
            del first
        else:
            form.launch(dz_last[:, m0:m1], x_rows, lower[m0:m1], dw[m0:m1], g)
    del lower  # its storage is dW's from here on
    if upper is not None:
        form.launch(dz_last[:, a:e], x_rows, upper, dw[a:e], g)
    del upper
    return dw


# a block-shifted label outside the recompute block (:func:`_dw_vocab_blocks`): equal to no column, and not negative
# (a negative label is an ignored row)
DW_BLOCK_FOREIGN_LABEL = 1 << 30


def _launch_stage(record, module_name: str, stage: str, values: dict, device) -> None:
    """One eager launch of ``stage`` on the caller's stream (:func:`_eager_launch`)."""
    geometry = Geometry.from_record(record)
    launch, arguments = _eager_launch(
        record, module_name, stage, values, device, geometry
    )
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _ffi_stream_context(int(index)):
        if launch.prepare is not None:
            launch.prepare(*arguments)
        launch.entry(*arguments)


def _x_rows_readable(X: torch.Tensor) -> bool:
    """Whether the GEMMs read rows of ``X`` in place (contiguous rows, 16-byte pitch and base)."""
    return X.stride(1) == 1 and X.stride(0) % 8 == 0 and X.data_ptr() % 16 == 0


def _dw_vocab_blocks(
    X: torch.Tensor,
    W: torch.Tensor,
    row_index: Optional[torch.Tensor],
    labels_c: torch.Tensor,
    lse_c: torch.Tensor,
    d_c: torch.Tensor,
    plan: Plan,
    grad: Optional[torch.Tensor],
    dw_out: torch.Tensor,
    *,
    v_start: int,
    block_rows: int,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    dz_last: Optional[torch.Tensor] = None,
    x_last: Optional[torch.Tensor] = None,
) -> None:
    """The weight-gradient rows ``[v_start, V)`` written into ``dw_out[v_start:]`` (BF16 ``[V, H]``) in vocabulary
    blocks of ``block_rows`` rows, from the row statistics of the call (the budget forms of :func:`memory_plan`).

    Per block ``[v0, v1)`` and per chunk of ``plan`` (in the plan's chunk order): the logits GEMM's ``dz`` arm
    (:data:`LOGPROB_DZ_STAGE`) on ``W[v0:v1]`` recomputes the chunk's ``dz`` block -- ``bf16(d_t * (1[v = y_t] -
    exp(z - lse_t)))``, the per-element sequence of ``row_grad`` on the same BF16-rounded logits, with the labels
    shifted into the block (a label of another block matches no column) -- and the weight-gradient accumulate GEMM adds
    ``dz_c^T @ X_c`` into the block's FP32 ``[v1 - v0, H]`` accumulator (the first chunk stores); the last chunk runs the
    fused scale + cast GEMM, ``bf16(g * (acc + dz_c^T @ X_c))`` into ``dw_out[v0:v1]`` (a one-chunk call: ``bf16(g *
    dz_c^T @ X_c)``).  Every element of ``dW`` therefore sees the FP32 operations of the chunk loop's own accumulate and
    cast in the same order: ``dW`` is bitwise the fast form's.  ``labels_c`` / ``lse_c`` / ``d_c`` are the plan's
    ``[rows]`` row vectors (compact when ``row_index`` is given; ``d`` the per-row gradient scale -- the loss entry's
    ``d_rows``, the log-probability entry's ``dlogp``), ``grad`` the upstream scalar (``None`` = 1).

    ``dz_last`` / ``x_last`` (the budget form ``"partial_ip"``: a deferred plan's last chunk, its BF16 ``[rows, V]``
    ``dz`` pinned since the forward and its BF16 ``X`` rows): the last chunk is not recomputed -- each block's cast reads
    the block's columns of ``dz_last`` (the loop's own ``dz``) and ``x_last``."""
    p = plan.problem
    H, V = int(p.hidden), int(p.vocab)
    # detached views (same storage): the row gather writes ``out=`` buffers, which a backward under
    # ``create_graph=True`` rejects for an input that requires a gradient; the recompute records no graph either way
    X, W = X.detach(), W.detach()
    device = X.device
    module_name, record = record_for(device)
    geometry = Geometry.from_record(record)
    arch = record["arch"]
    if LOGPROB_DZ_STAGE not in registered_stages(module_name):
        raise NotImplementedError(
            f"the registered program {module_name!r} lacks the stage {LOGPROB_DZ_STAGE!r} (memory_mode='budget')"
        )
    step = dw_block_step(geometry)
    v_start, block_rows = int(v_start), int(block_rows)
    if block_rows < step or block_rows % step or v_start % step or not 0 <= v_start < V:
        raise ValueError(
            f"_dw_vocab_blocks: v_start={v_start} / block_rows={block_rows} must be multiples of {step} below V={V}"
        )
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    dummy = _device_constants(index)["f32_dummy"]
    g = (
        _unit_scale(device)
        if grad is None
        else grad.detach().reshape(1).to(device=device, dtype=torch.float32)
    )
    chunks = plan.chunks
    n = len(chunks)
    pinned = dz_last is not None
    if pinned and (
        x_last is None
        or n < 2
        or tuple(dz_last.shape) != (int(chunks[-1][1]), V)
        or tuple(x_last.shape) != (int(chunks[-1][1]), H)
    ):
        raise ValueError(
            "_dw_vocab_blocks: the pinned last chunk needs a multi-chunk plan, its [rows, V] dz and [rows, H] X rows"
        )
    rows_max = max(rows_c for _, rows_c in chunks)
    T_loop = int(plan.rows)
    gather = row_index is not None or not _x_rows_readable(X)
    x_buf = (
        torch.empty((rows_max, H), dtype=torch.bfloat16, device=device)
        if gather
        else None
    )
    dw_tuning = _resolve_gemm_tuning(gemm_tuning).get("dw", {})

    def base_values() -> dict[str, Any]:
        values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
        values.update({name: 0 for name in COMMON_SCALARS})
        return values

    for v0 in range(v_start, V, block_rows):
        v1 = min(V, v0 + block_rows)
        vb = v1 - v0
        lb = torch.where(
            labels_c < 0,
            labels_c,
            torch.where(
                (labels_c >= v0) & (labels_c < v1),
                labels_c - v0,
                DW_BLOCK_FOREIGN_LABEL,
            ),
        ).contiguous()
        dz = torch.empty((rows_max, vb), dtype=torch.bfloat16, device=device)
        stats = torch.empty(
            (rows_max, -(-vb // int(geometry.stats_tile)), 2),
            dtype=torch.float32,
            device=device,
        )
        acc = (
            torch.empty((vb, H), dtype=torch.float32, device=device) if n > 1 else None
        )
        Wb = W[v0:v1]
        for i, (row0, rows_c) in enumerate(chunks):
            v = plan.variants_of(i)
            if pinned and i == n - 1:
                # the deferred chunk: its own ``dz`` columns of this block and its ``X`` rows, no recompute
                a_op, xc = dz_last[:, v0:v1], x_last
            else:
                a_op = None
            if a_op is None:
                if row_index is not None:
                    xc = torch.index_select(
                        X, 0, row_index[row0 : row0 + rows_c], out=x_buf[:rows_c]
                    )
                elif gather:
                    xc = x_buf[:rows_c]
                    xc.copy_(X[row0 : row0 + rows_c])
                else:
                    xc = X[row0 : row0 + rows_c]
                a_op = dz[:rows_c]
                values = base_values()
                values.update(
                    A=xc,
                    B=Wb,
                    C=a_op,
                    STATS_OUT=stats,
                    WS=dummy,
                    labels=lb,
                    lse=lse_c,
                    d_in=d_c,
                    rows_c=int(rows_c),
                    row0=int(row0),
                    d_off=int(row0),
                    T=T_loop,
                    H=H,
                    V=vb,
                    loss_div=1.0,
                    **gemm_scalars(
                        LOGPROB_DZ_STAGE,
                        geometry,
                        hidden=H,
                        vocab=vb,
                        rows_c=int(rows_c),
                        group_m=v.logits_group_m,
                    ),
                )
                _launch_stage(record, module_name, LOGPROB_DZ_STAGE, values, device)
            values = base_values()
            if i < n - 1:
                stage = plan.dw_acc_stage(i)
                values.update(
                    A=a_op,
                    B=xc,
                    C=acc,
                    STATS_OUT=stats,
                    WS=dummy,
                    rows_c=int(rows_c),
                    T=T_loop,
                    H=H,
                    V=vb,
                    loss_div=1.0,
                    **gemm_scalars(
                        stage,
                        geometry,
                        hidden=H,
                        vocab=vb,
                        rows_c=int(rows_c),
                        group_m=v.dw_group_m,
                        group_n=v.dw_group_n,
                        first_chunk=int(i == 0),
                    ),
                )
            else:
                # the cast GEMM of :func:`dw_cast` on this block (the same raster / store rules)
                raster = raster_variant(H, int(rows_c), arch)
                group_m = (
                    dw_tuning["group_m"]
                    if "group_m" in dw_tuning
                    else (raster[1] if raster else None)
                )
                stage = stage_variant(
                    "gemm_dw_cast_bf16",
                    blocked=False,
                    epi_store="tma"
                    if (
                        acc is not None
                        and "epi_store" not in dw_tuning
                        and dw_cast_tma_rule(2, int(rows_c), arch)
                    )
                    else None,
                )
                values.update(
                    A=a_op,
                    B=xc,
                    C=dw_out[v0:v1],
                    STATS_OUT=g,
                    WS=acc if acc is not None else dummy,
                    rows_c=int(rows_c),
                    T=T_loop,
                    H=H,
                    V=vb,
                    loss_div=1.0,
                    **gemm_scalars(
                        stage,
                        geometry,
                        hidden=H,
                        vocab=vb,
                        rows_c=int(rows_c),
                        group_m=group_m,
                        group_n=None,
                        first_chunk=int(acc is None),
                    ),
                )
            _launch_stage(record, module_name, stage, values, device)
        # the launch values hold views of this block's buffers: drop them before the next block allocates its own
        del values, xc, a_op, acc, dz, stats, lb, Wb


def _dw_vocab_blocks_temperature(
    X: torch.Tensor,
    W: torch.Tensor,
    row_index: Optional[torch.Tensor],
    labels_c: torch.Tensor,
    lse_c: torch.Tensor,
    d_c: torch.Tensor,
    plan: Plan,
    grad: Optional[torch.Tensor],
    dw_out: torch.Tensor,
    *,
    v_start: int,
    block_rows: int,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    temperature_rows: Optional[torch.Tensor] = None,
    dz_last: Optional[torch.Tensor] = None,
    x_last: Optional[torch.Tensor] = None,
) -> None:
    """:func:`_dw_vocab_blocks` of a call with a temperature (``temperature_rows``; ``dz_last`` / ``x_last``, the
    pinned last chunk of ``"partial_ip"``, as there).

    The weight-gradient rows ``[v_start, V)`` written into ``dw_out[v_start:]`` (BF16 ``[V, H]``) in vocabulary
    blocks of ``block_rows`` rows, from the row statistics of the call (the budget forms of :func:`memory_plan`).
    ``temperature_rows`` (the log-probability entry's compact FP32 ``[rows]`` temperature; the plan's problem must
    have one exactly when it is given): the recompute is the ``dz`` arm's temperature form
    (:data:`LOGPROB_DZ_TAU_STAGE`), the same per-element sequence as the chunk loop's.

    Per block ``[v0, v1)`` and per chunk of ``plan`` (in the plan's chunk order): the logits GEMM's ``dz`` arm
    (:data:`LOGPROB_DZ_STAGE`) on ``W[v0:v1]`` recomputes the chunk's ``dz`` block -- ``bf16(d_t * (1[v = y_t] -
    exp(z - lse_t)))``, the per-element sequence of ``row_grad`` on the same BF16-rounded logits, with the labels
    shifted into the block (a label of another block matches no column) -- and the weight-gradient accumulate GEMM adds
    ``dz_c^T @ X_c`` into the block's FP32 ``[v1 - v0, H]`` accumulator (the first chunk stores); the last chunk runs the
    fused scale + cast GEMM, ``bf16(g * (acc + dz_c^T @ X_c))`` into ``dw_out[v0:v1]`` (a one-chunk call: ``bf16(g *
    dz_c^T @ X_c)``).  Every element of ``dW`` therefore sees the FP32 operations of the chunk loop's own accumulate and
    cast in the same order: ``dW`` is bitwise the fast form's.  ``labels_c`` / ``lse_c`` / ``d_c`` are the plan's
    ``[rows]`` row vectors (compact when ``row_index`` is given; ``d`` the per-row gradient scale -- the loss entry's
    ``d_rows``, the log-probability entry's ``dlogp``), ``grad`` the upstream scalar (``None`` = 1)."""
    p = plan.problem
    H, V = int(p.hidden), int(p.vocab)
    # detached views (same storage): the row gather writes ``out=`` buffers, which a backward under
    # ``create_graph=True`` rejects for an input that requires a gradient; the recompute records no graph either way
    X, W = X.detach(), W.detach()
    device = X.device
    module_name, record = record_for(device)
    geometry = Geometry.from_record(record)
    arch = record["arch"]
    if p.temperature != (temperature_rows is not None):
        raise ValueError(
            "the budget recompute's temperature rows and its plan disagree (a call with a temperature recomputes "
            "with the dz arm's temperature form, a call without one with the plain dz arm)"
        )
    dz_stage = (
        LOGPROB_DZ_TAU_STAGE if temperature_rows is not None else LOGPROB_DZ_STAGE
    )
    if dz_stage not in registered_stages_with_temperature(module_name):
        raise NotImplementedError(
            f"the registered program {module_name!r} lacks the stage {dz_stage!r} (memory_mode='budget')"
        )
    step = dw_block_step(geometry)
    v_start, block_rows = int(v_start), int(block_rows)
    if block_rows < step or block_rows % step or v_start % step or not 0 <= v_start < V:
        raise ValueError(
            f"_dw_vocab_blocks: v_start={v_start} / block_rows={block_rows} must be multiples of {step} below V={V}"
        )
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    dummy = _device_constants(index)["f32_dummy"]
    g = (
        _unit_scale(device)
        if grad is None
        else grad.detach().reshape(1).to(device=device, dtype=torch.float32)
    )
    chunks = plan.chunks
    n = len(chunks)
    pinned = dz_last is not None
    if pinned and (
        x_last is None
        or n < 2
        or tuple(dz_last.shape) != (int(chunks[-1][1]), V)
        or tuple(x_last.shape) != (int(chunks[-1][1]), H)
    ):
        raise ValueError(
            "_dw_vocab_blocks: the pinned last chunk needs a multi-chunk plan, its [rows, V] dz and [rows, H] X rows"
        )
    rows_max = max(rows_c for _, rows_c in chunks)
    T_loop = int(plan.rows)
    gather = row_index is not None or not _x_rows_readable(X)
    x_buf = (
        torch.empty((rows_max, H), dtype=torch.bfloat16, device=device)
        if gather
        else None
    )
    dw_tuning = _resolve_gemm_tuning(gemm_tuning).get("dw", {})

    def base_values() -> dict[str, Any]:
        values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
        values.update({name: 0 for name in COMMON_SCALARS})
        return values

    for v0 in range(v_start, V, block_rows):
        v1 = min(V, v0 + block_rows)
        vb = v1 - v0
        lb = torch.where(
            labels_c < 0,
            labels_c,
            torch.where(
                (labels_c >= v0) & (labels_c < v1),
                labels_c - v0,
                DW_BLOCK_FOREIGN_LABEL,
            ),
        ).contiguous()
        dz = torch.empty((rows_max, vb), dtype=torch.bfloat16, device=device)
        stats = torch.empty(
            (rows_max, -(-vb // int(geometry.stats_tile)), 2),
            dtype=torch.float32,
            device=device,
        )
        acc = (
            torch.empty((vb, H), dtype=torch.float32, device=device) if n > 1 else None
        )
        Wb = W[v0:v1]
        for i, (row0, rows_c) in enumerate(chunks):
            v = plan.variants_of(i)
            if pinned and i == n - 1:
                # the deferred chunk: its own ``dz`` columns of this block and its ``X`` rows, no recompute
                a_op, xc = dz_last[:, v0:v1], x_last
            else:
                a_op = None
            if a_op is None:
                if row_index is not None:
                    xc = torch.index_select(
                        X, 0, row_index[row0 : row0 + rows_c], out=x_buf[:rows_c]
                    )
                elif gather:
                    xc = x_buf[:rows_c]
                    xc.copy_(X[row0 : row0 + rows_c])
                else:
                    xc = X[row0 : row0 + rows_c]
                a_op = dz[:rows_c]
                values = base_values()
                values.update(
                    A=xc,
                    B=Wb,
                    C=a_op,
                    STATS_OUT=stats,
                    WS=dummy,
                    labels=lb,
                    lse=lse_c,
                    d_in=d_c,
                    rows_c=int(rows_c),
                    row0=int(row0),
                    d_off=int(row0),
                    T=T_loop,
                    H=H,
                    V=vb,
                    loss_div=1.0,
                    **gemm_scalars(
                        dz_stage,
                        geometry,
                        hidden=H,
                        vocab=vb,
                        rows_c=int(rows_c),
                        group_m=v.logits_group_m,
                    ),
                )
                if temperature_rows is not None:
                    values.update(temperature=temperature_rows)  # read at row0 + r
                _launch_stage(record, module_name, dz_stage, values, device)
            values = base_values()
            if i < n - 1:
                stage = plan.dw_acc_stage(i)
                values.update(
                    A=a_op,
                    B=xc,
                    C=acc,
                    STATS_OUT=stats,
                    WS=dummy,
                    rows_c=int(rows_c),
                    T=T_loop,
                    H=H,
                    V=vb,
                    loss_div=1.0,
                    **gemm_scalars(
                        stage,
                        geometry,
                        hidden=H,
                        vocab=vb,
                        rows_c=int(rows_c),
                        group_m=v.dw_group_m,
                        group_n=v.dw_group_n,
                        first_chunk=int(i == 0),
                    ),
                )
            else:
                # the cast GEMM of :func:`dw_cast` on this block (the same raster / store rules)
                raster = raster_variant(H, int(rows_c), arch)
                group_m = (
                    dw_tuning["group_m"]
                    if "group_m" in dw_tuning
                    else (raster[1] if raster else None)
                )
                stage = stage_variant(
                    "gemm_dw_cast_bf16",
                    blocked=False,
                    epi_store="tma"
                    if (
                        acc is not None
                        and "epi_store" not in dw_tuning
                        and dw_cast_tma_rule(2, int(rows_c), arch)
                    )
                    else None,
                )
                values.update(
                    A=a_op,
                    B=xc,
                    C=dw_out[v0:v1],
                    STATS_OUT=g,
                    WS=acc if acc is not None else dummy,
                    rows_c=int(rows_c),
                    T=T_loop,
                    H=H,
                    V=vb,
                    loss_div=1.0,
                    **gemm_scalars(
                        stage,
                        geometry,
                        hidden=H,
                        vocab=vb,
                        rows_c=int(rows_c),
                        group_m=group_m,
                        group_n=None,
                        first_chunk=int(acc is None),
                    ),
                )
            _launch_stage(record, module_name, stage, values, device)
        # the launch values hold views of this block's buffers: drop them before the next block allocates its own
        del values, xc, a_op, acc, dz, stats, lb, Wb


def _dw_with_recompute(
    acc: list,
    grad: Optional[torch.Tensor],
    *,
    X: torch.Tensor,
    W: torch.Tensor,
    row_index: Optional[torch.Tensor],
    row_state: tuple,
    plan: Plan,
    dw_rows: int,
    block_rows: int,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
) -> torch.Tensor:
    """The BF16 ``dW`` of a budget form (:class:`MemoryPlan` ``"partial"`` / ``"recompute"``): the scaled cast of the
    chunk loop's FP32 accumulator of the rows ``[0, dw_rows)`` into the top rows of a new ``[V, H]`` tensor
    (:func:`scale_cast`, ``out=``), then the recompute of the other rows (:func:`_dw_vocab_blocks`).  ``acc`` is a
    one-element list holding the accumulator (or ``None``): it is taken out and dropped right after its cast, so the
    recompute's blocks never coexist with it when the caller holds no other reference.  ``row_state`` is
    ``(lse, d, labels)`` of the call's ``[rows]`` row vectors."""
    p = plan.problem
    V, H = int(p.vocab), int(p.hidden)
    dw = torch.empty((V, H), dtype=torch.bfloat16, device=X.device)
    accumulator = acc.pop() if acc else None
    if dw_rows:
        if accumulator is None or tuple(accumulator.shape) != (int(dw_rows), H):
            raise ValueError(
                f"the budget form needs the FP32 [{int(dw_rows)}, {H}] accumulator of its forward"
            )
        scale_cast(accumulator, grad, torch.bfloat16, out=dw[: int(dw_rows)])
    del accumulator
    lse_c, d_c, labels_c = row_state
    if int(dw_rows) < V:
        _dw_vocab_blocks(
            X,
            W,
            row_index,
            labels_c,
            lse_c,
            d_c,
            plan,
            grad,
            dw,
            v_start=int(dw_rows),
            block_rows=int(block_rows),
            gemm_tuning=gemm_tuning,
        )
    return dw


def _dw_with_recompute_temperature(
    acc: list,
    grad: Optional[torch.Tensor],
    *,
    X: torch.Tensor,
    W: torch.Tensor,
    row_index: Optional[torch.Tensor],
    row_state: tuple,
    plan: Plan,
    dw_rows: int,
    block_rows: int,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    temperature_rows: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """:func:`_dw_with_recompute` of a call with a temperature (``temperature_rows``).  The BF16 ``dW`` of a budget
    form (:class:`MemoryPlan` ``"partial"`` / ``"recompute"``): the scaled cast of the
    chunk loop's FP32 accumulator of the rows ``[0, dw_rows)`` into the top rows of a new ``[V, H]`` tensor
    (:func:`scale_cast`, ``out=``), then the recompute of the other rows (:func:`_dw_vocab_blocks`).  ``acc`` is a
    one-element list holding the accumulator (or ``None``): it is taken out and dropped right after its cast, so the
    recompute's blocks never coexist with it when the caller holds no other reference.  ``row_state`` is
    ``(lse, d, labels)`` of the call's ``[rows]`` row vectors; ``temperature_rows`` the log-probability entry's compact
    temperature (:func:`_dw_vocab_blocks`)."""
    p = plan.problem
    V, H = int(p.vocab), int(p.hidden)
    dw = torch.empty((V, H), dtype=torch.bfloat16, device=X.device)
    accumulator = acc.pop() if acc else None
    if dw_rows:
        if accumulator is None or tuple(accumulator.shape) != (int(dw_rows), H):
            raise ValueError(
                f"the budget form needs the FP32 [{int(dw_rows)}, {H}] accumulator of its forward"
            )
        scale_cast(accumulator, grad, torch.bfloat16, out=dw[: int(dw_rows)])
    del accumulator
    lse_c, d_c, labels_c = row_state
    if int(dw_rows) < V:
        _dw_vocab_blocks_temperature(
            X,
            W,
            row_index,
            labels_c,
            lse_c,
            d_c,
            plan,
            grad,
            dw,
            v_start=int(dw_rows),
            block_rows=int(block_rows),
            gemm_tuning=gemm_tuning,
            temperature_rows=temperature_rows,
        )
    return dw


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
    dw_acc_hi: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward of the loss entry from the saved accumulators: ``(dX, dW)`` = ``bf16(g * dX_acc)``,
    ``cast(g * dW_acc)``; the accumulators are read, never written (repeatable).  A compacted forward
    (``ForwardResult.row_index`` / ``num_rows``): the compact rows are cast once, then scattered into a
    zero BF16 ``[T, H]``.  A deferred weight gradient (``fuse_dw_cast``; ``ForwardResult.dz_last`` with
    ``x_last`` or ``x_src`` / ``x_idx``): ``dW = cast(g * (dW_acc + dz_last^T @ X_last))`` through the fused
    GEMM (:func:`dw_cast`), ``dW_acc`` being the chunks before the last one or ``None``;
    :meth:`ForwardResult.backward` passes everything.  The split accumulator of the budget form ``"fused_ip"``
    (``ForwardResult.dw_split`` set): ``dw_acc`` is the FP32 ``[a, H]`` sum of the vocabulary rows ``[0, a)`` and
    ``dw_acc_hi`` (``ForwardResult.dw_acc_hi``) the FP32 ``[V - a, H]`` sum of the rows ``[a, V)`` of the chunks before
    the last one; pass both with the deferred operands and ``dW`` is :func:`_dw_cast_split`'s out-of-place cast, bitwise
    :meth:`ForwardResult.backward`'s (``cake`` backend, BF16 ``dW``).  The form ``"partial_ip"`` and the other
    recompute forms (``ForwardResult.memory_plan.recompute``) recompute rows of ``dW`` from the forward's inputs:
    only :meth:`ForwardResult.backward` finishes them, and a split accumulator short of ``V`` rows is a
    ``ValueError`` here.  ``row_valid`` (``ForwardResult.row_valid``, the
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
    if dw_acc_hi is not None:
        # the split accumulator (``"fused_ip"``): checked before any launch
        if dz_last is None or dw_acc is None:
            raise ValueError(
                "dw_acc_hi (the split accumulator's upper rows) needs dw_acc (its lower rows) and the deferred "
                "operands dz_last with x_last or x_src / x_idx"
            )
        if backend != "cake" or grad_weight_dtype != torch.bfloat16:
            raise ValueError(
                f"the split accumulator (dw_acc_hi) of the budget forms {_SPLIT_FORMS} is cast by the cake kernels "
                "into a BF16 dW (backend='cake', grad_weight_dtype=torch.bfloat16)"
            )
        if int(dw_acc.shape[0]) + int(dw_acc_hi.shape[0]) != int(dz_last.shape[-1]):
            raise ValueError(
                "the split accumulator covers fewer than V rows (the budget form 'partial_ip' recomputes the "
                "others): finish it with ForwardResult.backward"
            )
    if dz_last is not None:
        if x_last is None:
            if x_src is None or x_idx is None:
                raise ValueError(
                    "a deferred weight gradient needs x_last, or x_src with the chunk's row index x_idx"
                )
            x_last = torch.index_select(
                x_src, 0, x_idx
            )  # the compacted chunk's rows, gathered again
        if dw_acc_hi is not None:
            return dx, _dw_cast_split(
                dz_last,
                x_last,
                [dw_acc, dw_acc_hi],
                grad,
                in_place=False,
                gemm_tuning=gemm_tuning,
            )
        if dw_acc is not None and int(dw_acc.shape[0]) != int(dz_last.shape[-1]):
            raise ValueError(
                "dw_acc holds fewer than V rows: a budget form's accumulator -- the split accumulator of "
                "'fused_ip' also needs its upper rows dw_acc_hi (ForwardResult.dw_acc_hi); the recompute forms "
                "finish through ForwardResult.backward"
            )
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
    num_valid_rows=None,
    check_num_valid_rows: bool = False,
) -> ForwardResult:
    """Forward of the log-probability entry: ``logp`` (FP32 ``[T]``, 0 on ignored rows) plus the saved statistic
    ``lse`` (compacted: the ``[T_v]`` rows of ``row_index``, exactly what :func:`backward_logprob` expects back,
    together with ``row_index``).  ``compact_rows`` (default :func:`compact_rows_default`), ``hidden_count`` (default
    :func:`hidden_count_default`) and ``gemm_tuning`` (default :func:`gemm_tuning_default`) as in
    :func:`forward_loss`; ``logprob_lbl`` (default :func:`logprob_lbl_default`) writes only the row statistics and
    the label column's logit from the logits GEMM (bitwise the same ``lse`` / ``logp``).  This forward has no
    weight gradient, so it takes no ``fuse_dw_cast`` / ``dx_finalize`` / ``dw_stream``.  ``num_valid_rows`` /
    ``check_num_valid_rows`` as in :func:`forward_loss` (the caller's count: no readback; with the flag one readback,
    compacted or not -- the log-probability entry's backward reuses the checked count)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
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
    given = None if num_valid_rows is None else _given_count(num_valid_rows, T)
    if T == 0:
        return _empty_forward(problem, X, need_dx=False, need_dw=False)
    compact = _resolve_compact(compact_rows)
    if given is None:
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
    else:
        # the caller's count: no readback, the index is formed on the device
        hc = None
        row_index = (
            given_valid_rows(labels, given, check=check_num_valid_rows)[0]
            if compact
            else None
        )
        if check_num_valid_rows and not compact:
            _check_count_on_host(labels, given)
    if row_index is not None and row_index.numel() == 0:  # every row ignored
        return _empty_forward(
            problem, X, need_dx=False, need_dw=False, row_index=row_index
        )
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
            return ForwardResult(
                loss=None,
                logp=scatter_rows(logp, row_index, T),
                lse=lse,
                memory=binding.memory,
                backend=backend,
                row_index=row_index,
                num_rows=T,
            )
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
        _template_only=backend == "cake" and hc is not None,
    )
    if hc is not None:
        # the hidden valid-row count: chunk 0's logits already sit in the prologue's buffers, so the first call launches
        # through the templated binding too (the runner's own workspace is never written)
        binding = _Binding.from_runner(runner, hc)
        del runner
        if key is not None:
            cache.remember(key, binding)
        logp, lse = binding.forward(X, W, labels, None, None, row_index, hc)
        memory = binding.memory
    else:
        logp, lse = runner.forward()
        if key is not None:
            cache.remember(key, _Binding.from_runner(runner))
        memory = runner.memory
    return ForwardResult(
        loss=None,
        logp=scatter_rows(logp, row_index, T),
        lse=lse,
        memory=memory,
        backend=backend,
        row_index=row_index,
        num_rows=T,
    )


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
    num_valid_rows=None,
    check_num_valid_rows: bool = False,
    dw_rows: Optional[int] = None,
    dw_split: Optional[int] = None,
    dw_split_end: Optional[int] = None,
) -> tuple[Optional[torch.Tensor], Any]:
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
    ``num_valid_rows`` / ``check_num_valid_rows`` (as in :func:`forward_loss`): the caller's count of valid rows, no
    readback; the recount path plans on it.  ``check_num_valid_rows=True`` checks a given count once on every path
    (with the forward's index or without gradients too).  ``dw_rows`` (``cake`` backend, ``fuse_dw_cast=False``;
    :attr:`Plan.dw_rows`): the chunk loop accumulates the weight gradient of the vocabulary rows ``[0, dw_rows)`` only
    and the second result is that FP32 ``[dw_rows, H]`` accumulator, uncast (the budget forms of
    :func:`memory_plan`).  ``dw_split`` / ``dw_split_end`` (``cake`` backend, ``fuse_dw_cast=True``, two or more chunks;
    :attr:`Plan.dw_split`, :attr:`Plan.dw_split_end`, the budget forms ``"fused_ip"`` / ``"partial_ip"``): the chunk
    loop accumulates the vocabulary rows ``[0, dw_split_end)`` into the split accumulator and the deferred last chunk
    runs no weight-gradient GEMM; the second result is the :class:`_SplitDw` of the caller's cast
    (:func:`_dw_cast_split`)."""
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
        if check_num_valid_rows and num_valid_rows is not None:
            _check_count_on_host(labels, _given_count(num_valid_rows, problem.num_rows))
        return None, None
    T, H, V = problem.num_rows, problem.hidden, problem.vocab
    fused_dx = bool(
        need_dx and _resolve_dx_finalize(dx_finalize)
    )  # resolved once per call
    compact = bool(T and _resolve_compact(compact_rows))
    hc = None
    if row_index is not _RECOUNT:
        # the forward's valid-row index: the plan is known without a count (the valid-row mask of the fused dX finalize
        # is one device comparison, no synchronization)
        if row_index is not None and not compact:
            raise ValueError("a valid-row index belongs to a compacted backward")
        row_valid = (labels >= 0) if (row_index is not None and fused_dx) else None
        if check_num_valid_rows and num_valid_rows is not None:
            _check_count_on_host(labels, _given_count(num_valid_rows, T))
    elif num_valid_rows is not None:
        # the caller's count: no readback (the index is formed on the device)
        given = _given_count(num_valid_rows, T)
        row_index, row_valid = (
            given_valid_rows(labels, given, mask=fused_dx, check=check_num_valid_rows)
            if compact
            else (None, None)
        )
        if check_num_valid_rows and not compact:
            _check_count_on_host(labels, given)
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
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    if dw_rows is not None and (backend != "cake" or not need_dw or fuse):
        raise ValueError(
            "dw_rows belongs to a cake-backend log-probability backward with an unfused weight gradient "
            "(need_dw=True, fuse_dw_cast=False)"
        )
    if dw_split is not None and (backend != "cake" or not fuse or dw_rows is not None):
        raise ValueError(
            "dw_split belongs to a cake-backend log-probability backward with the fused weight-gradient cast "
            "(need_dw=True, fuse_dw_cast=True, no dw_rows)"
        )
    elif dw_split is None and dw_split_end is not None:
        raise ValueError("dw_split_end belongs to a backward with dw_split")
    if rows == 0:  # no rows, or every row ignored
        return (
            torch.zeros((T, H), dtype=torch.bfloat16, device=X.device)
            if need_dx
            else None,
            (
                torch.zeros((V, H), dtype=torch.bfloat16, device=X.device)
                if dw_rows is None
                else torch.zeros(
                    (int(dw_rows), H), dtype=torch.float32, device=X.device
                )
            )
            if need_dw
            else None,
        )
    _check_output(lse, "lse", (rows,), torch.float32)
    dlogp = dlogp.detach().reshape(T).to(torch.float32).contiguous()
    cache = BINDING_CACHE
    key = None
    valid_count = None if row_index is None else rows
    # the fused dX finalize of a compacted backward: the runner leaves dx_acc uncast and finalize_dx writes the [T, H]
    # dX in one pass (uncompacted: the in-loop flat cast is already one pass over every row)
    dx_cast = not (fused_dx and row_index is not None)

    def finish_dx(dx):
        if dx is None:
            return None
        if dx_cast:
            return scatter_rows(dx, row_index, T)
        return finalize_dx(dx, None, row_index, row_valid, T, backend=backend)

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
        )
        if dw_rows is not None:
            key += (
                int(dw_rows),
            )  # a budget form's accumulated rows (:attr:`Plan.dw_rows`)
        if dw_split is not None:
            # a split accumulator's row and end (:attr:`Plan.dw_split`)
            key += ("dw_split", int(dw_split), dw_split_end)
        binding = cache.lookup(key)
        if binding is not None:
            dx, dw = binding.backward_logprob(
                X, W, labels, lse, dlogp, row_index, hc, stream_mode
            )
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
        dw_rows=dw_rows,
        dw_split=dw_split,
        dw_split_end=dw_split_end,
        # a split accumulator: the per-call buffers of a binding also on the first call (the deferred chunk's ``dz``
        # pins its chunk buffer only, not the runner's whole workspace)
        _template_only=backend == "cake" and (hc is not None or dw_split is not None),
    )
    if hc is not None or dw_split is not None:
        binding = _Binding.from_runner(runner, hc)
        del runner
        if key is not None:
            cache.remember(key, binding)
        dx, dw = binding.backward_logprob(
            X, W, labels, lse, dlogp, row_index, hc, stream_mode
        )
        return finish_dx(dx), dw
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
    num_valid_rows=None,
    check_num_valid_rows: bool = False,
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
    weight gradient, so it takes no ``fuse_dw_cast`` / ``dx_finalize`` / ``dw_stream``.  ``num_valid_rows`` /
    ``check_num_valid_rows`` as in :func:`forward_loss` (the caller's count: no readback; with the flag one readback,
    compacted or not -- the log-probability entry's backward reuses the checked count).

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
    given = None if num_valid_rows is None else _given_count(num_valid_rows, T)
    if T == 0:
        return _empty_forward(problem, X, need_dx=False, need_dw=False)
    compact = _resolve_compact(compact_rows)
    if given is None:
        hc = _hidden_count_temperature(
            problem,
            X,
            W,
            labels,
            backend=backend,
            compact=compact,
            stats=True,
            hidden_count=_resolve_hidden_count(hidden_count),
            temperature=tau,
        )
        if hc is not None:
            row_index = hc.row_index()
        else:
            row_index = valid_row_index(labels) if compact else None
    else:
        # the caller's count: no readback, the index is formed on the device
        hc = None
        row_index = (
            given_valid_rows(labels, given, check=check_num_valid_rows)[0]
            if compact
            else None
        )
        if check_num_valid_rows and not compact:
            _check_count_on_host(labels, given)
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
        _template_only=backend == "cake" and hc is not None,
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
    num_valid_rows=None,
    check_num_valid_rows: bool = False,
    dw_rows: Optional[int] = None,
    temperature=None,
    check_temperature: bool = False,
    temperature_rows: Optional[torch.Tensor] = None,
    dw_split: Optional[int] = None,
    dw_split_end: Optional[int] = None,
) -> tuple[Optional[torch.Tensor], Any]:
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
    ``num_valid_rows`` / ``check_num_valid_rows`` (as in :func:`forward_loss`): the caller's count of valid rows, no
    readback; the recount path plans on it.  ``check_num_valid_rows=True`` checks a given count once on every path
    (with the forward's index or without gradients too).  ``dw_rows`` (``cake`` backend, ``fuse_dw_cast=False``;
    :attr:`Plan.dw_rows`): the chunk loop accumulates the weight gradient of the vocabulary rows ``[0, dw_rows)`` only
    and the second result is that FP32 ``[dw_rows, H]`` accumulator, uncast (the budget forms of
    :func:`memory_plan`).

    Temperature (the forward's): ``temperature_rows`` -- the forward's compact FP32 ``[rows]`` temperature
    (:attr:`ForwardResult.temperature_rows`, the rows of ``row_index``; the autograd Function passes it) -- or
    ``temperature`` / ``check_temperature`` as given to :func:`forward_logprob` (validated again and gathered by the
    valid-row index; a recount with a temperature takes the host count).  ``dz = (dlogp_t / tau_t) * (1[v = y_t] -
    softmax(z_t / tau_t))`` with one FP32 division for ``dlogp_t / tau_t`` per row and one per logit."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_det_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        entry="logprob",
    )
    if temperature_rows is not None and temperature is not None:
        raise ValueError(
            "pass the forward's temperature_rows or the call's temperature, not both"
        )
    tau = (
        None
        if temperature_rows is not None
        else _resolve_temperature(
            temperature, problem.num_rows, X.device, check=check_temperature
        )
    )
    has_tau = temperature_rows is not None or tau is not None
    _check_temperature_arms(has_tau, logprob_dz=logprob_dz)
    if has_tau:
        problem = _with_temperature(problem)
    stream_mode = dw_stream_mode(dw_stream)
    tuning = _resolve_gemm_tuning(gemm_tuning)
    dz_arm = _resolve_logprob_dz(logprob_dz)
    if not (need_dx or need_dw):
        if check_num_valid_rows and num_valid_rows is not None:
            _check_count_on_host(labels, _given_count(num_valid_rows, problem.num_rows))
        return None, None
    T, H, V = problem.num_rows, problem.hidden, problem.vocab
    fused_dx = bool(
        need_dx and _resolve_dx_finalize(dx_finalize)
    )  # resolved once per call
    compact = bool(T and _resolve_compact(compact_rows))
    hc = None
    if row_index is not _RECOUNT:
        # the forward's valid-row index: the plan is known without a count (the valid-row mask of the fused dX finalize
        # is one device comparison, no synchronization)
        if row_index is not None and not compact:
            raise ValueError("a valid-row index belongs to a compacted backward")
        row_valid = (labels >= 0) if (row_index is not None and fused_dx) else None
        if check_num_valid_rows and num_valid_rows is not None:
            _check_count_on_host(labels, _given_count(num_valid_rows, T))
    elif num_valid_rows is not None:
        # the caller's count: no readback (the index is formed on the device)
        given = _given_count(num_valid_rows, T)
        row_index, row_valid = (
            given_valid_rows(labels, given, mask=fused_dx, check=check_num_valid_rows)
            if compact
            else (None, None)
        )
        if check_num_valid_rows and not compact:
            _check_count_on_host(labels, given)
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
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    if dw_rows is not None and (backend != "cake" or not need_dw or fuse):
        raise ValueError(
            "dw_rows belongs to a cake-backend log-probability backward with an unfused weight gradient "
            "(need_dw=True, fuse_dw_cast=False)"
        )
    if dw_split is not None and (backend != "cake" or not fuse or dw_rows is not None):
        raise ValueError(
            "dw_split belongs to a cake-backend log-probability backward with the fused weight-gradient cast "
            "(need_dw=True, fuse_dw_cast=True, no dw_rows)"
        )
    elif dw_split is None and dw_split_end is not None:
        raise ValueError("dw_split_end belongs to a backward with dw_split")
    if rows == 0:  # no rows, or every row ignored
        return (
            torch.zeros((T, H), dtype=torch.bfloat16, device=X.device)
            if need_dx
            else None,
            (
                torch.zeros((V, H), dtype=torch.bfloat16, device=X.device)
                if dw_rows is None
                else torch.zeros(
                    (int(dw_rows), H), dtype=torch.float32, device=X.device
                )
            )
            if need_dw
            else None,
        )
    _check_output(lse, "lse", (rows,), torch.float32)
    tau_rows: Optional[torch.Tensor] = None  # the call without a temperature makes none
    if temperature_rows is not None:
        # the forward's compact temperature (checked like ``lse``: the same rows)
        _check_output(temperature_rows, "temperature_rows", (rows,), torch.float32)
        if temperature_rows.device != X.device:
            raise ValueError("temperature_rows must be on X's device")
        tau_rows = temperature_rows
    elif tau is not None:
        tau_rows = _temperature_rows(tau, row_index, rows, X.device)
    dlogp = dlogp.detach().reshape(T).to(torch.float32).contiguous()
    cache = BINDING_CACHE
    key = None
    valid_count = None if row_index is None else rows
    # the fused dX finalize of a compacted backward: the runner leaves dx_acc uncast and finalize_dx writes the [T, H]
    # dX in one pass (uncompacted: the in-loop flat cast is already one pass over every row)
    dx_cast = not (fused_dx and row_index is not None)

    def finish_dx(dx):
        if dx is None:
            return None
        if dx_cast:
            return scatter_rows(dx, row_index, T)
        return finalize_dx(dx, None, row_index, row_valid, T, backend=backend)

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
        )
        if dw_rows is not None:
            key += (
                int(dw_rows),
            )  # a budget form's accumulated rows (:attr:`Plan.dw_rows`)
        if dw_split is not None:
            # a split accumulator's row and end (:attr:`Plan.dw_split`)
            key += ("dw_split", int(dw_split), dw_split_end)
        if has_tau:
            key = temperature_binding_key(key)
        binding = cache.lookup(key)
        if binding is not None:
            dx, dw = (
                cast(_TemperatureBinding, binding).backward_logprob(
                    X,
                    W,
                    labels,
                    lse,
                    dlogp,
                    row_index,
                    hc,
                    stream_mode,
                    temperature=tau_rows,
                )
                if has_tau
                else binding.backward_logprob(
                    X, W, labels, lse, dlogp, row_index, hc, stream_mode
                )
            )
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
        dw_rows=dw_rows,
        temperature=tau_rows,
        dw_split=dw_split,
        dw_split_end=dw_split_end,
        # a split accumulator: the per-call buffers of a binding also on the first call (the deferred chunk's ``dz``
        # pins its chunk buffer only, not the runner's whole workspace)
        _template_only=backend == "cake" and (hc is not None or dw_split is not None),
    )
    if hc is not None:
        binding = _TemperatureBinding.from_runner(runner, hc)
        del runner
        if key is not None:
            cache.remember(key, binding)
        dx, dw = binding.backward_logprob(
            X, W, labels, lse, dlogp, row_index, hc, stream_mode
        )
        return finish_dx(dx), dw
    if dw_split is not None:
        binding = _TemperatureBinding.from_runner(runner)
        del runner
        if key is not None:
            cache.remember(key, binding)
        dx, dw = (
            cast(_TemperatureBinding, binding).backward_logprob(
                X,
                W,
                labels,
                lse,
                dlogp,
                row_index,
                None,
                stream_mode,
                temperature=tau_rows,
            )
            if has_tau
            else binding.backward_logprob(
                X, W, labels, lse, dlogp, row_index, None, stream_mode
            )
        )
        return finish_dx(dx), dw
    dx, dw = runner.backward()
    if key is not None:
        cache.remember(key, _TemperatureBinding.from_runner(runner))
    # the runner's chunk workspace is not needed by the dX finalize: dropped before its output is allocated
    del runner
    return finish_dx(dx), dw


_CHECK_WITHOUT_COUNT = (
    "check_num_valid_rows=True needs num_valid_rows (the count it checks)"
)

_SECOND_BACKWARD = (
    "Trying to backward through the graph a second time: the chunked LM-head entry released its saved state in the "
    "previous backward (specify retain_graph=True to backward through the graph again)"
)


class CakeLMHeadDetLossFunction(torch.autograd.Function):
    """Entry (a): the forward produces the FP32 gradient accumulators; the backward scales and casts them.

    With ``memory_mode="budget"``, or a default call (:data:`MEMORY_AUTO`) whose automatic form is a budget form
    (``ctx.early_release``; a checkpointed call learns it from its recomputation), a backward that frees the graph (no
    ``retain_graph`` / ``create_graph``; :func:`_keep_graph`) releases the saved state as soon as it is used -- the
    ``dX`` accumulator right after ``dX`` is written, before the weight gradient's buffers exist, the ``dW``
    accumulator and the row state when the backward ends -- instead of when the graph is freed; another backward
    through the same graph then raises.  Otherwise, and in a retained graph, it keeps everything until the graph is
    freed (the backward is repeatable, the accumulators are read, never written)."""

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
        memory_mode,
        num_valid_rows,
        check_num_valid_rows,
    ):
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        # gradient placement "auto" inside a non-reentrant torch.utils.checkpoint (as in :mod:`.cake_backend`): the
        # checkpointed forward computes the loss only and saves the inputs; the recomputation runs this forward and
        # hands its result to the checkpointed call's backward.  Outside one (the plain step): the fused forward
        ckpt = None
        if _main._TOP_SAVED_HOOKS(False) is not None and (
            need_dx or need_dw or _main._selective_checkpoint_modes()
        ):
            # under a selective checkpoint's modes (any need_dx / need_dw): again, with every dispatch mode popped (as
            # in :mod:`.cake_backend`)
            if _main._selective_checkpoint_modes():
                with _main._modes_off():
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
                        memory_mode,
                        num_valid_rows,
                        check_num_valid_rows,
                    )
            CakeLMHeadDetLossFunction._setup(
                ctx, X, grad_weight_dtype, backend, memory_mode
            )
            ckpt = _main._checkpoint_pass()
        if ckpt is not None:
            options = dict(
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
                memory_mode=memory_mode,
                num_valid_rows=num_valid_rows,
                check_num_valid_rows=check_num_valid_rows,
                return_logp=bool(return_logp),
            )
            split, handed = _main._loss_checkpoint_split(
                ctx,
                ckpt,
                X,
                W,
                labels,
                infer_logp,
                loss_weights,
                need_dx,
                need_dw,
                options,
            )
            if split:
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
                    _main._hand_over(ctx, ckpt, handed, result)
                ctx.mark_non_differentiable(result.logp)
                return result.loss, result.logp
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
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
            return_logp=bool(return_logp),
        )
        CakeLMHeadDetLossFunction._setup(
            ctx, X, grad_weight_dtype, backend, memory_mode
        )
        # the deferred weight gradient's operands: the last chunk's dz rows and its X rows, the latter a view of the
        # input X (or X itself with the chunk's row index when compacted) -- saved through the autograd context so an
        # in-place write to X between the forward and the backward raises PyTorch's saved-tensor version error
        # instead of yielding a stale dW.  ``memory_mode="budget"``: the picked form (``None`` = the fast form); a
        # recompute form also saves X, W and labels likewise (an uncompacted call's row-state labels are the caller's
        # tensor itself) and keeps the row state and plan its recompute reads
        if result.memory_plan is None or not result.memory_plan.recompute:
            ctx.save_for_backward(
                result.dz_last, result.x_last, result.x_src, result.x_idx
            )
        else:
            ctx.save_for_backward(
                result.dz_last,
                result.x_last,
                result.x_src,
                result.x_idx,
                X,
                W,
                labels,
            )
        CakeLMHeadDetLossFunction._keep(ctx, result)
        ctx.mark_non_differentiable(result.logp)
        return result.loss, result.logp

    @staticmethod
    def _setup(ctx, X, grad_weight_dtype, backend, memory_mode) -> None:
        """The call's own backward state (every forward form)."""
        ctx.set_materialize_grads(False)
        ctx.num_rows = int(X.shape[0])
        ctx.grad_weight_dtype = grad_weight_dtype
        ctx.backend = backend
        ctx.released = False
        # the early release of the saved state: the budget forms only (their peak model assumes it) -- ``"budget"``,
        # or the automatic form once its forward picked one (:meth:`_keep`)
        mode = _resolve_memory_mode(memory_mode)
        ctx.early_release = mode == "budget"
        ctx.auto_release = mode == MEMORY_AUTO
        ctx.split = None

    @staticmethod
    def _keep(ctx, result) -> None:
        """The gradient state the backward reads, from a fused forward's result."""
        ctx.dx_acc, ctx.dw_acc = (
            result.dx_acc,
            result.dw_acc,
        )  # saved state, never mutated
        ctx.memory_plan = mplan = result.memory_plan
        # a split accumulator (``"fused_ip"`` / ``"partial_ip"``): its rows and upper part (``dw_acc`` the lower one)
        ctx.dw_split, ctx.dw_acc_hi = result.dw_split, result.dw_acc_hi
        if ctx.auto_release:
            # the automatic form: the early release of the budget form its forward picked (none: the fast form)
            ctx.early_release = mplan is not None
        if mplan is not None and mplan.recompute:
            ctx.row_state, ctx.plan = result.row_state, result.plan
        ctx.row_index = (
            result.row_index
        )  # compacted: dx_acc holds the rows of row_index
        ctx.row_valid = (
            result.row_valid
        )  # the fused dX finalize's valid-row mask (None otherwise)
        # the forward's resolved GEMM knobs: the deferred weight-gradient GEMM takes them
        ctx.gemm_tuning = result.gemm_tuning
        ctx.empty = ctx.num_rows == 0 or (
            result.row_index is not None and result.row_index.numel() == 0
        )

    @staticmethod
    def _release(ctx) -> None:
        ctx.dx_acc = ctx.dw_acc = ctx.dw_acc_hi = ctx.dw_split = None
        ctx.row_state = ctx.plan = None
        ctx.row_index = ctx.row_valid = None
        ctx.released = True

    @staticmethod
    def backward(ctx, grad_loss, grad_logp=None):
        none = (None,) * 20
        if grad_loss is None:
            return none
        if ctx.released:
            raise RuntimeError(_SECOND_BACKWARD)
        saved = None
        if ctx.split is not None:
            # a checkpointed call: unpacking its inputs runs the recomputation, whose fused forward's result (or this
            # backward's own fused forward) is the gradient state
            inputs = ctx.saved_tensors
            state = _main._split_state(ctx, inputs, forward_loss)
            CakeLMHeadDetLossFunction._keep(ctx, state)
            saved = (state.dz_last, state.x_last, state.x_src, state.x_idx) + (
                tuple(inputs[:2])
                if state.memory_plan is not None and state.memory_plan.recompute
                else ()
            )
            # the backward reads the state through ``ctx`` and ``saved`` from here on: a local reference would keep
            # the accumulators the early release drops alive until the backward returns
            state = inputs = None
        # after the recomputation of a checkpointed call, whose form sets the automatic form's early release
        keep = not ctx.early_release or _keep_graph()
        if not keep and ctx.split is not None and ctx.split.result is not None:
            # a recomputed call's own state, read once: released like the rest
            ctx.split = ctx.split._replace(result=None)
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
            if not keep:
                CakeLMHeadDetLossFunction._release(ctx)
            return (dx, dw) + none[2:]
        mplan = ctx.memory_plan
        if saved is None:
            saved = ctx.saved_tensors
        if mplan is not None and mplan.recompute:
            # (the saved labels only version-check the row state's labels; a checkpointed call saves its inputs)
            dz_last, x_last, x_src, x_idx, X, W = saved[:6]
        else:
            mplan = None  # the fast form, or a budget form without a recompute: the weight gradient as before
            dz_last, x_last, x_src, x_idx = saved
        # the operations of :func:`backward_loss`, with the saved state released as soon as it is used
        dx = None
        if ctx.dx_acc is not None:
            dx = finalize_dx(
                ctx.dx_acc,
                grad_loss,
                ctx.row_index,
                ctx.row_valid,
                ctx.num_rows,
                backend=ctx.backend,
            )
            if not keep:
                ctx.dx_acc = None  # released before the weight gradient's buffers exist
        dw = None
        if ctx.dw_split is not None and dz_last is not None:
            if x_last is None:
                x_last = torch.index_select(
                    x_src, 0, x_idx
                )  # the compacted chunk's rows, gathered again
            # a split accumulator (``"fused_ip"`` / ``"partial_ip"``): a backward that frees the graph writes ``dW``
            # into the lower accumulator's storage (the accumulators released from the context first, so ``dW`` is
            # the storage's only holder)
            acc = [ctx.dw_acc, ctx.dw_acc_hi]
            if not keep:
                ctx.dw_acc = ctx.dw_acc_hi = None
            dw = _dw_cast_split(
                dz_last,
                x_last,
                acc,
                grad_loss,
                in_place=not keep,
                gemm_tuning=ctx.gemm_tuning,
            )
            del acc
            if mplan is not None:
                # ``"partial_ip"``: the rows above the accumulated ones, recomputed for the chunks before the last;
                # the deferred chunk's own ``dz`` columns feed each block's cast
                lse_c, d_c, labels_c = ctx.row_state
                _dw_vocab_blocks(
                    X,
                    W,
                    ctx.row_index,
                    labels_c,
                    lse_c,
                    d_c,
                    ctx.plan,
                    grad_loss,
                    dw,
                    v_start=int(ctx.dw_split[1]),
                    block_rows=int(mplan.block_rows),
                    gemm_tuning=ctx.gemm_tuning,
                    dz_last=dz_last,
                    x_last=x_last,
                )
        elif mplan is not None:
            acc = [] if ctx.dw_acc is None else [ctx.dw_acc]
            if not keep:
                ctx.dw_acc = (
                    None  # released right after its cast, before the recompute's blocks
                )
            dw = _dw_with_recompute(
                acc,
                grad_loss,
                X=X,
                W=W,
                row_index=ctx.row_index,
                row_state=ctx.row_state,
                plan=ctx.plan,
                dw_rows=mplan.dw_rows,
                block_rows=mplan.block_rows,
                gemm_tuning=ctx.gemm_tuning,
            )
        elif dz_last is not None:
            if x_last is None:
                x_last = torch.index_select(
                    x_src, 0, x_idx
                )  # the compacted chunk's rows, gathered again
            dw = dw_cast(
                dz_last,
                x_last,
                ctx.dw_acc,
                grad_loss,
                ctx.grad_weight_dtype,
                backend=ctx.backend,
                gemm_tuning=ctx.gemm_tuning,
            )
        elif ctx.dw_acc is not None:
            dw = scale_cast(
                ctx.dw_acc, grad_loss, ctx.grad_weight_dtype, backend=ctx.backend
            )
        if not keep:
            CakeLMHeadDetLossFunction._release(ctx)
        return (dx, dw) + none[2:]


def _logprob_call_key(
    inputs: tuple,
    labels: torch.Tensor,
    chunk_size,
    backend,
    compact: bool,
    hidden: bool,
    tuning,
    lbl_arm: bool,
    mode,
    num_valid_rows,
    check_num_valid_rows,
    *extra,
) -> tuple:
    """What a recomputed call of this module's log-probability functions must match to take its checkpointed call's
    row results (:func:`.cake_backend._logprob_call_key` over this module's options, the memory mode and the caller's
    count among them; ``extra``: the temperature's)."""
    return _main._logprob_call_key(
        inputs,
        (
            "opt_in",
            chunk_size,
            backend,
            compact,
            hidden,
            _main._tuning_key(tuning),
            lbl_arm,
            mode,
            _given_count(num_valid_rows, int(labels.numel())),
            bool(check_num_valid_rows),
        )
        + extra,
    )


class CakeLMHeadDetLogprobFunction(torch.autograd.Function):
    """Entry (b): the forward saves the row statistic; the backward recomputes the logits.  Inside a non-reentrant
    ``torch.utils.checkpoint`` the recomputation takes the checkpointed call's ``logp`` and row results instead of
    running the logits forward again (as :class:`.cake_backend.CakeLMHeadDetLogprobFunction`)."""

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
        memory_mode,
        num_valid_rows,
        check_num_valid_rows,
    ):
        if (
            _main._TOP_SAVED_HOOKS(False) is not None
            and _main._selective_checkpoint_modes()
        ):
            with _main._modes_off():  # under a selective checkpoint's modes: popped (as in :mod:`.cake_backend`)
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
                    memory_mode,
                    num_valid_rows,
                    check_num_valid_rows,
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
        mode = None if memory_mode is None else _resolve_memory_mode(memory_mode)
        budget = mode == "budget"
        if budget:
            _check_budget_call(backend, torch.bfloat16)
        # the automatic form (a default call): its backward picks the fast form or a budget form
        auto = _logprob_auto(mode, backend, X, W)
        lbl_arm = _resolve_logprob_lbl(logprob_lbl)
        # inside a non-reentrant checkpoint: the recomputation takes the checkpointed call's row results (as
        # :mod:`.cake_backend`'s entry; the same frame record, see :class:`.cake_backend._LogprobKept`)
        ckpt = (
            _main._checkpoint_pass()
            if _main._TOP_SAVED_HOOKS(False) is not None
            else None
        )
        kept = None
        if ckpt is not None:
            inputs = (X, W, labels)
            key = _logprob_call_key(
                inputs,
                labels,
                chunk_size,
                backend,
                compact,
                hidden,
                tuning,
                lbl_arm,
                mode,
                num_valid_rows,
                check_num_valid_rows,
            )
            if not ckpt.first:
                kept = _main._logprob_checkpoint_take(ckpt, key, inputs)
        if kept is not None:
            # the recomputation of a checkpointed call: its row results, no logits GEMM
            logp, (lse, row_index) = kept.logp.clone(), kept.saved
        else:
            result = forward_logprob(
                X,
                W,
                labels,
                chunk_size=chunk_size,
                backend=backend,
                compact_rows=compact,
                # ``"budget"``: no hidden valid-row count (as :func:`forward_loss`); the automatic form keeps it where
                # its chunk-0 buffers fit the budget (:func:`_auto_hidden_count`)
                hidden_count=(
                    False
                    if budget
                    else (
                        _auto_hidden_count(
                            hidden,
                            X,
                            W,
                            chunk_size,
                            need_dx=bool(ctx.needs_input_grad[0]),
                            entry="logprob",
                        )
                        if auto and ctx.needs_input_grad[1]
                        else hidden
                    )
                ),
                gemm_tuning=tuning,
                logprob_lbl=lbl_arm,
                num_valid_rows=num_valid_rows,
                check_num_valid_rows=check_num_valid_rows,
            )
            logp, lse, row_index = result.logp, result.lse, result.row_index
            if ckpt is not None and ckpt.first:
                _main._logprob_checkpoint_keep(
                    ctx, ckpt, key, inputs, logp, (lse, row_index)
                )
        ctx.set_materialize_grads(False)
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
        ctx.budget, ctx.auto = budget, auto
        # a budget form: the caller's count sizes the backward's budget of an uncompacted call (the forward checked it
        # with ``check_num_valid_rows``)
        ctx.num_valid_rows = (
            _given_count(num_valid_rows, int(labels.numel()))
            if budget or auto
            else None
        )
        return logp

    @staticmethod
    def backward(ctx, dlogp):
        if dlogp is None:
            return (None,) * 16
        X, W, labels, lse, row_index = ctx.saved_tensors
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        mplan = (
            _auto_logprob_plan(
                X,
                W,
                labels,
                row_index,
                compact=ctx.compact,
                given=ctx.num_valid_rows,
                need_dx=need_dx,
                fuse_dw_cast=ctx.fuse_dw_cast,
                dx_finalize=ctx.dx_finalize,
                chunk=ctx.chunk_size,
                tuning=ctx.gemm_tuning,
                backend=ctx.backend,
            )
            if ctx.auto and need_dw
            else None
        )
        if need_dw and (ctx.budget or mplan is not None):
            dx, dw = _backward_logprob_budget(
                X,
                W,
                labels,
                lse,
                dlogp,
                need_dx=need_dx,
                fuse_dw_cast=ctx.fuse_dw_cast,
                chunk_size=ctx.chunk_size,
                backend=ctx.backend,
                compact_rows=ctx.compact,
                dx_finalize=ctx.dx_finalize,
                dw_stream=ctx.dw_stream,
                hidden_count=ctx.hidden_count,
                gemm_tuning=ctx.gemm_tuning,
                logprob_dz=ctx.logprob_dz,
                row_index=row_index,
                num_valid_rows=ctx.num_valid_rows,
                mplan=mplan,
            )
        else:
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
            )
        return (dx, dw) + (None,) * 14


class CakeLMHeadDetLogprobTemperatureFunction(torch.autograd.Function):
    """:class:`CakeLMHeadDetLogprobFunction` of a call with a temperature (:func:`cake_lm_head_det_train_logprob`),
    with the same checkpoint hand-over."""

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
        memory_mode,
        num_valid_rows,
        check_num_valid_rows,
        temperature,
        check_temperature,
    ):
        if (
            _main._TOP_SAVED_HOOKS(False) is not None
            and _main._selective_checkpoint_modes()
        ):
            with _main._modes_off():  # under a selective checkpoint's modes: popped (as in :mod:`.cake_backend`)
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
                    memory_mode,
                    num_valid_rows,
                    check_num_valid_rows,
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
        mode = None if memory_mode is None else _resolve_memory_mode(memory_mode)
        budget = mode == "budget"
        if budget:
            _check_budget_call(backend, torch.bfloat16)
        # the automatic form (a default call): its backward picks the fast form or a budget form
        auto = _logprob_auto(mode, backend, X, W)
        lbl_arm = _resolve_logprob_lbl(logprob_lbl)
        # inside a non-reentrant checkpoint: the recomputation takes the checkpointed call's row results (as
        # :class:`CakeLMHeadDetLogprobFunction`)
        ckpt = (
            _main._checkpoint_pass()
            if _main._TOP_SAVED_HOOKS(False) is not None
            else None
        )
        kept = None
        if ckpt is not None:
            tau_in = temperature if isinstance(temperature, torch.Tensor) else None
            inputs = (X, W, labels, tau_in)
            key = _logprob_call_key(
                inputs,
                labels,
                chunk_size,
                backend,
                compact,
                hidden,
                tuning,
                lbl_arm,
                mode,
                num_valid_rows,
                check_num_valid_rows,
                logprob_dz,
                None if tau_in is not None else temperature,
                bool(check_temperature),
            )
            if not ckpt.first:
                kept = _main._logprob_checkpoint_take(ckpt, key, inputs)
        if kept is not None:
            # the recomputation of a checkpointed call: its row results, no logits GEMM
            logp, saved = kept.logp.clone(), kept.saved
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
                # ``"budget"``: no hidden valid-row count (as :func:`forward_loss`); the automatic form keeps it where
                # its chunk-0 buffers fit the budget (:func:`_auto_hidden_count`)
                hidden_count=(
                    False
                    if budget
                    else (
                        _auto_hidden_count(
                            hidden,
                            X,
                            W,
                            chunk_size,
                            need_dx=bool(ctx.needs_input_grad[0]),
                            entry="logprob",
                        )
                        if auto and ctx.needs_input_grad[1]
                        else hidden
                    )
                ),
                gemm_tuning=tuning,
                logprob_lbl=lbl_arm,
                num_valid_rows=num_valid_rows,
                check_num_valid_rows=check_num_valid_rows,
                temperature=tau,
            )
            logp = result.logp
            saved = (result.lse, result.row_index, result.temperature_rows)
            if ckpt is not None and ckpt.first:
                _main._logprob_checkpoint_keep(ctx, ckpt, key, inputs, logp, saved)
        ctx.set_materialize_grads(False)
        # the valid-row index rides along: the backward plans its chunks without counting the rows again; so does the
        # compact temperature the forward's kernels read (no second gather)
        ctx.save_for_backward(X, W, labels, *saved)
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
        ctx.budget, ctx.auto = budget, auto
        # a budget form: the caller's count sizes the backward's budget of an uncompacted call (the forward checked it
        # with ``check_num_valid_rows``)
        ctx.num_valid_rows = (
            _given_count(num_valid_rows, int(labels.numel()))
            if budget or auto
            else None
        )
        return logp

    @staticmethod
    def backward(ctx, dlogp):
        if dlogp is None:
            return (None,) * 18
        X, W, labels, lse, row_index, tau_rows = ctx.saved_tensors
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        mplan = (
            _auto_logprob_plan(
                X,
                W,
                labels,
                row_index,
                compact=ctx.compact,
                given=ctx.num_valid_rows,
                need_dx=need_dx,
                fuse_dw_cast=ctx.fuse_dw_cast,
                dx_finalize=ctx.dx_finalize,
                chunk=ctx.chunk_size,
                tuning=ctx.gemm_tuning,
                backend=ctx.backend,
                temperature=tau_rows is not None,
            )
            if ctx.auto and need_dw
            else None
        )
        if need_dw and (ctx.budget or mplan is not None):
            dx, dw = _backward_logprob_budget_temperature(
                X,
                W,
                labels,
                lse,
                dlogp,
                need_dx=need_dx,
                fuse_dw_cast=ctx.fuse_dw_cast,
                chunk_size=ctx.chunk_size,
                backend=ctx.backend,
                compact_rows=ctx.compact,
                dx_finalize=ctx.dx_finalize,
                dw_stream=ctx.dw_stream,
                hidden_count=ctx.hidden_count,
                gemm_tuning=ctx.gemm_tuning,
                logprob_dz=ctx.logprob_dz,
                row_index=row_index,
                num_valid_rows=ctx.num_valid_rows,
                mplan=mplan,
                temperature_rows=tau_rows,
            )
        else:
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
                temperature_rows=tau_rows,
            )
        return (dx, dw) + (None,) * 16


class CakeLMHeadDetLogprobAutoFunction(_main.CakeLMHeadDetLogprobFunction):
    """The default call of the log-probability entry without the caller's count (:data:`MEMORY_AUTO`):
    :class:`.cake_backend.CakeLMHeadDetLogprobFunction`'s forward (the hidden valid-row count, a checkpoint's kept row
    results) and, where the fast form fits the budget with its own lifetimes, its backward (the zero-gradient row skip
    included); else the budget form its backward picks (:func:`_auto_logprob_plan`, :func:`_backward_logprob_budget`;
    no zero-gradient row skip there)."""

    @staticmethod
    def backward(ctx, dlogp):
        if dlogp is None:
            return (None,) * 13
        # unpacked once: a second unpack inside a non-reentrant checkpoint raises
        saved = ctx.saved_tensors
        X, W, labels, lse, row_index = saved
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        mplan = (
            _auto_logprob_plan(
                X,
                W,
                labels,
                row_index,
                compact=ctx.compact,
                given=None,
                need_dx=need_dx,
                fuse_dw_cast=ctx.fuse_dw_cast,
                dx_finalize=ctx.dx_finalize,
                chunk=ctx.chunk_size,
                tuning=ctx.gemm_tuning,
                backend=ctx.backend,
            )
            if need_dw
            else None
        )
        if mplan is None:
            dx, dw = _main._logprob_backward(ctx, dlogp, saved)
        else:
            dx, dw = _backward_logprob_budget(
                X,
                W,
                labels,
                lse,
                dlogp,
                need_dx=need_dx,
                fuse_dw_cast=ctx.fuse_dw_cast,
                chunk_size=ctx.chunk_size,
                backend=ctx.backend,
                compact_rows=ctx.compact,
                dx_finalize=ctx.dx_finalize,
                dw_stream=ctx.dw_stream,
                hidden_count=ctx.hidden_count,
                gemm_tuning=ctx.gemm_tuning,
                logprob_dz=ctx.logprob_dz,
                row_index=row_index,
                mplan=mplan,
            )
        return (dx, dw) + (None,) * 11


class CakeLMHeadDetLogprobTemperatureAutoFunction(
    _main.CakeLMHeadDetLogprobTemperatureFunction
):
    """:class:`CakeLMHeadDetLogprobAutoFunction` of a call with a temperature: the forward of
    :class:`.cake_backend.CakeLMHeadDetLogprobTemperatureFunction`, its backward where the fast form fits, else the
    budget form (:func:`_backward_logprob_budget_temperature` with the forward's compact temperature;
    :func:`_backward_logprob_budget` when the forward kept none)."""

    @staticmethod
    def backward(ctx, dlogp):
        if dlogp is None:
            return (None,) * 15
        # unpacked once: a second unpack inside a non-reentrant checkpoint raises
        saved = ctx.saved_tensors
        X, W, labels, lse, row_index, tau_rows, _ = saved
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        mplan = (
            _auto_logprob_plan(
                X,
                W,
                labels,
                row_index,
                compact=ctx.compact,
                given=None,
                need_dx=need_dx,
                fuse_dw_cast=ctx.fuse_dw_cast,
                dx_finalize=ctx.dx_finalize,
                chunk=ctx.chunk_size,
                tuning=ctx.gemm_tuning,
                backend=ctx.backend,
                temperature=tau_rows is not None,
            )
            if need_dw
            else None
        )
        if mplan is None:
            dx, dw = _main._logprob_temperature_backward(ctx, dlogp, saved)
            return (dx, dw) + (None,) * 13
        common = dict(
            need_dx=need_dx,
            fuse_dw_cast=ctx.fuse_dw_cast,
            chunk_size=ctx.chunk_size,
            backend=ctx.backend,
            compact_rows=ctx.compact,
            dx_finalize=ctx.dx_finalize,
            dw_stream=ctx.dw_stream,
            hidden_count=ctx.hidden_count,
            gemm_tuning=ctx.gemm_tuning,
            logprob_dz=ctx.logprob_dz,
            row_index=row_index,
            mplan=mplan,
        )
        if tau_rows is None:
            dx, dw = _backward_logprob_budget(X, W, labels, lse, dlogp, **common)
        else:
            dx, dw = _backward_logprob_budget_temperature(
                X, W, labels, lse, dlogp, temperature_rows=tau_rows, **common
            )
        return (dx, dw) + (None,) * 13


def _backward_logprob_budget(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    *,
    need_dx: bool,
    fuse_dw_cast: bool,
    row_index: Optional[torch.Tensor],
    num_valid_rows: Optional[int] = None,
    mplan: Optional[MemoryPlan] = None,
    **common,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """:func:`backward_logprob` (with ``dW``) in the form :func:`memory_plan` picks for ``memory_mode="budget"``
    against the budget of the valid rows (``row_index``'s, else the caller's ``num_valid_rows`` -- checked by the
    forward with ``check_num_valid_rows`` --, else one host count of an uncompacted call; :func:`_budget_valid_rows`):
    ``"fused"`` -- the fast form; ``"fused_ip"`` / ``"partial_ip"`` (two or more chunks; one chunk runs the fused
    form) -- the chunk loop accumulates the rows ``[0, dw_rows)`` into the split accumulator (:attr:`Plan.dw_split`)
    and keeps the deferred last chunk's ``dz``, whose fused cast writes ``dW`` into the lower accumulator's storage
    (:func:`_dw_cast_split`), and ``"partial_ip"`` recomputes the other rows of the chunks before the last
    (:func:`_dw_vocab_blocks` with the pinned last chunk); ``"unfused"`` -- the chunk loop accumulates every row and
    the FP32 accumulator is cast after ``dX`` is written; ``"partial"`` / ``"recompute"`` -- the chunk loop
    accumulates the rows ``[0, dw_rows)`` (none for ``"recompute"``), their cast fills the top rows of ``dW`` and the
    other rows are recomputed in vocabulary blocks from the saved ``lse`` and ``dlogp`` (:func:`_dw_vocab_blocks`).
    ``mplan``: the form a default call's automatic form picked (:func:`_auto_logprob_plan`), which this backward runs
    instead of planning.  Bitwise the fast form's ``dX`` / ``dW``."""
    T, H, V = int(X.shape[0]), int(W.shape[1]), int(W.shape[0])
    rows = T if row_index is None else int(row_index.numel())
    chunk = int(common["chunk_size"])
    if rows == 0:
        return backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            row_index=row_index,
            **{k: v for k, v in common.items() if k != "row_index"},
        )
    problem = validate_lm_head_det_inputs(
        X, W, labels, chunk_size=chunk, entry="logprob"
    )
    mplan = (
        mplan
        if mplan is not None
        else _budget_plan(
            "logprob",
            X.device,
            num_rows=T,
            rows=rows,
            hidden=H,
            vocab=V,
            chunk=chunk,
            compact=row_index is not None,
            need_dx=need_dx,
            fuse_dw_cast=bool(fuse_dw_cast),
            buffer_rows=min(chunk, rows),
            x_copy=problem.x_copy,
            dx_finalize=_resolve_dx_finalize(common["dx_finalize"]),
            budget_rows=_budget_valid_rows(
                labels,
                row_index,
                num_valid_rows,
                bool(common["compact_rows"]),
            ),
            tuning=common["gemm_tuning"],
        )
    )
    loop = dict(common, row_index=row_index)
    if mplan.form in _SPLIT_FORMS and -(-rows // chunk) < 2:
        # one chunk: no accumulator to split (the fused form)
        mplan = replace(mplan, form="fused", dw_rows=V, block_rows=None)
    if mplan.form == "fused":
        return backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            fuse_dw_cast=True,
            **loop,
        )
    if mplan.form in _SPLIT_FORMS:
        geometry = Geometry.from_record(record_for(X.device)[1])
        end = V if mplan.form == "fused_ip" else int(mplan.dw_rows)
        dx, split = backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            fuse_dw_cast=True,
            dw_split=dw_split_rows(V, geometry),
            dw_split_end=end,
            **loop,
        )
        return dx, _split_dw(
            split,
            mplan,
            end,
            X=X,
            W=W,
            labels=labels,
            lse=lse,
            dlogp=dlogp,
            row_index=row_index,
            problem=problem,
            geometry=geometry,
            gemm_tuning=common["gemm_tuning"],
        )
    if mplan.form == "unfused":
        dx, acc = backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            fuse_dw_cast=False,
            dw_rows=V,
            **loop,
        )
        held = [acc]
        del acc
        return dx, _dw_cast_taken(held)
    v0 = int(mplan.dw_rows)
    acc_list: list = []
    dx = None
    if v0:
        dx, acc = backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            fuse_dw_cast=False,
            dw_rows=v0,
            **loop,
        )
        acc_list.append(acc)
        del acc
    elif need_dx:
        dx, _ = backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=True,
            need_dw=False,
            fuse_dw_cast=False,
            **loop,
        )
    module_name, record = record_for(X.device)
    geometry = Geometry.from_record(record)
    plan = make_plan(
        problem,
        need_dx=False,
        need_dw=True,
        geometry=geometry,
        num_sms=int(torch.cuda.get_device_properties(X.device).multi_processor_count),
        valid_rows=None if row_index is None else rows,
        arch=record["arch"],
        gemm_tuning=common["gemm_tuning"],
    )
    d = dlogp.detach().reshape(T).to(torch.float32).contiguous()
    dw = _dw_with_recompute(
        acc_list,
        None,
        X=X,
        W=W,
        row_index=row_index,
        row_state=(
            lse,
            d if row_index is None else d.index_select(0, row_index),
            _prepare_labels(labels, geometry, row_index),
        ),
        plan=plan,
        dw_rows=v0,
        block_rows=int(mplan.block_rows),
        gemm_tuning=common["gemm_tuning"],
    )
    return dx, dw


def _backward_logprob_budget_temperature(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    *,
    need_dx: bool,
    fuse_dw_cast: bool,
    row_index: Optional[torch.Tensor],
    num_valid_rows: Optional[int] = None,
    mplan: Optional[MemoryPlan] = None,
    **common,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """:func:`_backward_logprob_budget` of a call with a temperature (``temperature_rows`` in ``common``; ``mplan`` as
    there).
    :func:`backward_logprob` (with ``dW``) in the form :func:`memory_plan` picks for ``memory_mode="budget"``
    against the budget of the valid rows (``row_index``'s, else the caller's ``num_valid_rows`` -- checked by the
    forward with ``check_num_valid_rows`` --, else one host count of an uncompacted call; :func:`_budget_valid_rows`):
    ``"fused"`` -- the fast form; ``"fused_ip"`` / ``"partial_ip"`` -- the split accumulator and the pinned last chunk
    as in :func:`_backward_logprob_budget`; ``"unfused"`` -- the chunk loop accumulates every row and the FP32
    accumulator is cast after ``dX`` is written; ``"partial"`` / ``"recompute"`` -- the chunk loop accumulates the rows
    ``[0, dw_rows)`` (none for ``"recompute"``), their cast fills the top rows of ``dW`` and the other rows are
    recomputed in vocabulary blocks from the saved ``lse`` and ``dlogp`` (:func:`_dw_vocab_blocks`).  Bitwise the fast
    form's ``dX`` / ``dW``."""
    T, H, V = int(X.shape[0]), int(W.shape[1]), int(W.shape[0])
    rows = T if row_index is None else int(row_index.numel())
    chunk = int(common["chunk_size"])
    if rows == 0:
        return backward_logprob_temperature(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            row_index=row_index,
            **{k: v for k, v in common.items() if k != "row_index"},
        )
    tau_rows = common.get("temperature_rows")
    # the recompute's plan carries the call's temperature (its dz recompute is the temperature form)
    problem = validate_lm_head_det_inputs(
        X,
        W,
        labels,
        chunk_size=chunk,
        entry="logprob",
    )
    if tau_rows is not None:
        problem = _with_temperature(problem)
    mplan = (
        mplan
        if mplan is not None
        else (
            _budget_plan_temperature
            if common.get("temperature_rows") is not None
            else _budget_plan
        )(
            "logprob",
            X.device,
            num_rows=T,
            rows=rows,
            hidden=H,
            vocab=V,
            chunk=chunk,
            compact=row_index is not None,
            need_dx=need_dx,
            fuse_dw_cast=bool(fuse_dw_cast),
            buffer_rows=min(chunk, rows),
            x_copy=problem.x_copy,
            dx_finalize=_resolve_dx_finalize(common["dx_finalize"]),
            budget_rows=_budget_valid_rows(
                labels,
                row_index,
                num_valid_rows,
                bool(common["compact_rows"]),
            ),
            tuning=common["gemm_tuning"],
        )
    )
    loop = dict(common, row_index=row_index)
    if mplan.form in _SPLIT_FORMS and -(-rows // chunk) < 2:
        # one chunk: no accumulator to split (the fused form)
        mplan = replace(mplan, form="fused", dw_rows=V, block_rows=None)
    if mplan.form == "fused":
        return backward_logprob_temperature(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            fuse_dw_cast=True,
            **loop,
        )
    if mplan.form in _SPLIT_FORMS:
        geometry = Geometry.from_record(record_for(X.device)[1])
        end = V if mplan.form == "fused_ip" else int(mplan.dw_rows)
        dx, split = backward_logprob_temperature(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            fuse_dw_cast=True,
            dw_split=dw_split_rows(V, geometry),
            dw_split_end=end,
            **loop,
        )
        return dx, _split_dw(
            split,
            mplan,
            end,
            X=X,
            W=W,
            labels=labels,
            lse=lse,
            dlogp=dlogp,
            row_index=row_index,
            problem=problem,
            geometry=geometry,
            gemm_tuning=common["gemm_tuning"],
            temperature_rows=tau_rows,
        )
    if mplan.form == "unfused":
        dx, acc = backward_logprob_temperature(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            fuse_dw_cast=False,
            dw_rows=V,
            **loop,
        )
        held = [acc]
        del acc
        return dx, _dw_cast_taken(held)
    v0 = int(mplan.dw_rows)
    acc_list: list = []
    dx = None
    if v0:
        dx, acc = backward_logprob_temperature(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=need_dx,
            need_dw=True,
            fuse_dw_cast=False,
            dw_rows=v0,
            **loop,
        )
        acc_list.append(acc)
        del acc
    elif need_dx:
        dx, _ = backward_logprob_temperature(
            X,
            W,
            labels,
            lse,
            dlogp,
            need_dx=True,
            need_dw=False,
            fuse_dw_cast=False,
            **loop,
        )
    module_name, record = record_for(X.device)
    geometry = Geometry.from_record(record)
    plan = make_plan(
        problem,
        need_dx=False,
        need_dw=True,
        geometry=geometry,
        num_sms=int(torch.cuda.get_device_properties(X.device).multi_processor_count),
        valid_rows=None if row_index is None else rows,
        arch=record["arch"],
        gemm_tuning=common["gemm_tuning"],
    )
    d = dlogp.detach().reshape(T).to(torch.float32).contiguous()
    dw = _dw_with_recompute_temperature(
        acc_list,
        None,
        X=X,
        W=W,
        row_index=row_index,
        row_state=(
            lse,
            d if row_index is None else d.index_select(0, row_index),
            _prepare_labels(labels, geometry, row_index),
        ),
        plan=plan,
        dw_rows=v0,
        block_rows=int(mplan.block_rows),
        gemm_tuning=common["gemm_tuning"],
        temperature_rows=tau_rows,
    )
    return dx, dw


# the budget forms with a split accumulator (:attr:`Plan.dw_split`)
_SPLIT_FORMS = ("fused_ip", "partial_ip")


def _split_dw(
    split: _SplitDw,
    mplan: MemoryPlan,
    end: int,
    *,
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    row_index: Optional[torch.Tensor],
    problem: Problem,
    geometry: Geometry,
    gemm_tuning: Optional[dict[str, dict[str, Any]]],
    temperature_rows: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The BF16 ``dW`` of a log-probability backward with a split accumulator (``"fused_ip"`` / ``"partial_ip"``):
    the deferred last chunk's fused cast into the lower accumulator's storage (:func:`_dw_cast_split`, in place: the
    accumulators are this backward's own), then for ``"partial_ip"`` (``end < V``) the recompute of the rows ``[end,
    V)`` of the chunks before the last, the last chunk's block read from its pinned ``dz``
    (:func:`_dw_vocab_blocks`; with ``temperature_rows``, :func:`_dw_vocab_blocks_temperature`).  ``problem``: the
    call's (with its temperature for a call with one)."""
    T, V = int(X.shape[0]), int(W.shape[0])
    x_last = split.x_rows()
    dz_last, acc = split.dz_last, split.acc
    del split
    dw = _dw_cast_split(
        dz_last, x_last, acc, None, in_place=True, gemm_tuning=gemm_tuning
    )
    if end < V:
        rows = T if row_index is None else int(row_index.numel())
        module_name, record = record_for(X.device)
        plan = make_plan(
            problem,
            need_dx=False,
            need_dw=True,
            geometry=geometry,
            num_sms=int(
                torch.cuda.get_device_properties(X.device).multi_processor_count
            ),
            valid_rows=None if row_index is None else rows,
            arch=record["arch"],
            gemm_tuning=gemm_tuning,
        )
        d = dlogp.detach().reshape(T).to(torch.float32).contiguous()
        d_c = d if row_index is None else d.index_select(0, row_index)
        labels_c = _prepare_labels(labels, geometry, row_index)
        tail = dict(
            v_start=int(end),
            block_rows=int(mplan.block_rows),
            gemm_tuning=gemm_tuning,
            dz_last=dz_last,
            x_last=x_last,
        )
        if temperature_rows is None and not problem.temperature:
            _dw_vocab_blocks(
                X, W, row_index, labels_c, lse, d_c, plan, None, dw, **tail
            )
        else:
            _dw_vocab_blocks_temperature(
                X,
                W,
                row_index,
                labels_c,
                lse,
                d_c,
                plan,
                None,
                dw,
                temperature_rows=temperature_rows,
                **tail,
            )
    return dw


def _dw_cast_taken(acc: list) -> torch.Tensor:
    """The BF16 cast of the FP32 accumulator held (alone) by the one-element list ``acc``, which is emptied: the
    accumulator is freed as soon as its cast is queued."""
    accumulator = acc.pop()
    return scale_cast(accumulator, None, torch.bfloat16)


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
    memory_mode: Optional[str] = None,
    num_valid_rows=None,
    check_num_valid_rows: bool = False,
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
    * ``memory_mode`` (default ``None``: the automatic form,
      :func:`memory_mode_default`): ``"fast"`` keeps the FP32 ``[V, H]``
      weight-gradient accumulator next to the chunk buffers; ``"budget"``
      bounds the step peak by the bytes of a head that materializes the BF16
      logits of the valid rows (:func:`memory_plan`): per call, the fastest
      exact weight-gradient form that fits -- the fused or the unfused cast,
      or an accumulate of the leading vocabulary rows plus a recompute of the
      others in vocabulary blocks in the backward (bitwise the same ``dW``,
      more GEMM work).  ``cake`` backend only.  The default runs the fast form
      where it fits that bound with its own buffer lifetimes, else the budget
      form that fits (:func:`auto_memory_plan`; none: the fast form) -- on the
      ``cake`` backend with a CUDA ``X`` and a BF16 ``W`` whose gradient is
      needed; every other default call runs the fast form.

    Two keywords that are not switches:

    * ``num_valid_rows`` (default ``None``): the number of labels ``>= 0`` as a
      host integer (Python / NumPy integer or a CPU scalar tensor).  The
      forward then plans on it without waiting for the device; the count is
      checked on the device (an asynchronous assertion: a wrong count ends the
      CUDA context with a device-side assertion).  ``check_num_valid_rows=True``
      reads the count back instead and raises ``ValueError`` on a mismatch (one
      synchronization; for debugging).  An uncompacted call forms no index from
      the count (an uncompacted call of a budget form, explicit or automatic,
      sizes its budget on it; an uncompacted default call without it keeps the
      fast form); only
      ``check_num_valid_rows=True`` checks it there, with the same readback and
      ``ValueError``.  With the flag every call checks its count once, in the
      forward, whatever ``compact_rows``, ``memory_mode`` or the inputs that
      need a gradient.  ``None``: the forward counts the valid rows itself (one
      synchronization per call).

    ``check_num_valid_rows=True`` without ``num_valid_rows`` is a
    ``ValueError``.

    Without ``num_valid_rows``, with ``memory_mode="fast"`` or a default call
    whose step forms no BF16 weight gradient on the ``cake`` backend (another
    backend, ``X`` off CUDA, an FP32 ``W``, a ``W`` that requires no
    gradient), the call is :mod:`.cake_backend`'s
    ``cake_lm_head_det_train_loss``.  A default call under gradient mode off
    whose ``W`` requires a gradient (e.g. a reentrant checkpoint's first
    pass) computes the loss here in the fast form, its hidden valid-row count
    checked against the step's budget (:func:`_auto_hidden_count`; the same
    bits).  Its gradient placement
    (:func:`.cake_backend.set_gradient_placement`) applies here too.  With
    ``memory_mode="budget"``, or a default call whose automatic form is a
    budget form, a backward that
    frees the graph releases the forward's saved accumulators as soon as it
    has used them; a second backward through the same graph then raises
    (``retain_graph=True`` keeps them: the backward stays repeatable).
    """
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_CHECK_WITHOUT_COUNT)
    mode = _resolve_memory_mode(memory_mode)
    if num_valid_rows is None and (
        mode == "fast" or (mode == MEMORY_AUTO and not _auto_reaches(backend, X, W))
    ):
        return _main.cake_lm_head_det_train_loss(
            X,
            W,
            labels,
            objective=objective,
            loss_div=loss_div,
            infer_logp=infer_logp,
            loss_weights=loss_weights,
            chunk_size=chunk_size,
            return_logp=return_logp,
            grad_weight_dtype=grad_weight_dtype,
            backend=backend,
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
        )
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
        and _main.gradient_placement() == "auto"
    ):
        # no autograd graph follows (as in :mod:`.cake_backend`): the loss (and logp) only, no gradient GEMM
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
            memory_mode=mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
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
        mode,
        num_valid_rows,
        check_num_valid_rows,
    )
    return (loss, logp.detach()) if return_logp else loss


def _logprob_keep_given(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int,
    compact_rows: Optional[bool],
    fuse_dw_cast: Optional[bool],
    dx_finalize: Optional[bool],
    dw_stream: Optional[Union[bool, str]],
    hidden_count: Optional[bool],
    gemm_tuning: Optional[dict[str, dict[str, Any]]],
    logprob_dz: Optional[bool],
    logprob_lbl: Optional[bool],
    num_valid_rows,
    check_num_valid_rows: bool,
) -> torch.Tensor:
    """``memory_mode="keep_logits"`` with the caller's valid-row count ``num_valid_rows`` (``cake`` backend): where a
    backward can follow, :mod:`.cake_backend`'s kept-logits function over the valid-row index formed on the device
    from the count (:func:`given_valid_rows`: no readback; ``check_num_valid_rows`` reads it back once), so the
    forward counts nothing; elsewhere the default entry with the same count (as the kept entry's own fallback).
    Bitwise the same ``logp`` / ``dX`` / ``dW`` as the kept entry without the count."""
    if not (torch.is_grad_enabled() and (X.requires_grad or W.requires_grad)):
        return CakeLMHeadDetLogprobFunction.apply(
            X,
            W,
            labels,
            int(chunk_size),
            "cake",
            compact_rows,
            fuse_dw_cast,
            dx_finalize,
            dw_stream,
            hidden_count,
            gemm_tuning,
            logprob_dz,
            logprob_lbl,
            _resolve_memory_mode(None),
            num_valid_rows,
            check_num_valid_rows,
        )
    problem = validate_lm_head_det_inputs(
        X, W, labels, chunk_size=chunk_size, entry="logprob"
    )
    T = int(problem.num_rows)
    given = _given_count(num_valid_rows, T)
    row_index = None
    if T and _resolve_compact(compact_rows):
        row_index = given_valid_rows(labels, given, check=check_num_valid_rows)[0]
    elif T and check_num_valid_rows:
        _check_count_on_host(labels, given)
    return _main.CakeLMHeadDetLogprobKeepFunction.apply(
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
        row_index,
    )


# The one-chunk kept logits of the default log-probability call (:func:`set_logprob_one_chunk_keep`, on by default;
# the rule is :func:`_one_chunk_keep_rule`): a call whose valid rows fit one chunk keeps their BF16 logits -- at most
# one chunk, the chunk bound of the memory rule -- from its forward to its backward.
_ONE_CHUNK_KEEP: dict[str, bool] = {"on": True}


def set_logprob_one_chunk_keep(on: bool) -> bool:
    """Process switch of the default log-probability call's one-chunk kept logits (default ``True``; read once per
    call) and return the previous setting.  On: a default call that :func:`_one_chunk_keep_rule` keeps runs the kept
    form of :mod:`.cake_backend` -- the forward keeps the BF16 logits of the valid rows (``2 * T_v * V`` bytes, at
    most one chunk) until the backward, which forms ``dz`` from them instead of recomputing them (three GEMMs instead
    of four; bitwise the same ``logp`` / ``dX`` / ``dW``).  ``False``: every default call runs as with
    ``memory_mode="fast"`` (which turns the route off for one call)."""
    previous = _ONE_CHUNK_KEEP["on"]
    _ONE_CHUNK_KEEP["on"] = bool(on)
    return previous


def _one_chunk_keep_rule(
    X: Any,
    W: Any,
    labels: Any,
    chunk_size: Any,
    backend: str,
    compact_rows: Optional[bool],
    dx_finalize: Optional[bool],
    logprob_dz: Optional[bool],
) -> Optional[Callable[[Optional[torch.Tensor]], bool]]:
    """The one rule of the one-chunk kept logits (:func:`set_logprob_one_chunk_keep`) for a default
    log-probability call -- the automatic memory form, no ``num_valid_rows``, no temperature: ``None`` when the call
    keeps its route, else the predicate ``keep(row_index)`` that the call's forward applies to its valid-row index
    right after counting it (:class:`CakeLMHeadDetLogprobOneChunkFunction`, :func:`.cake_backend.forward_logprob`).

    Admitted (a predicate): the switch on, the ``cake`` backend, ``X`` on a CUDA device, a backward that can follow
    (grad mode on, ``X`` or ``W`` trainable), no saved-tensor hooks on top (neither pass of a non-reentrant
    checkpoint, no CPU offload), outside a backward (a reentrant checkpoint's recomputation) and fewer than
    :data:`.cake_backend.HIDDEN_COUNT_MIN_CHUNKS` chunks of rows, whose forward counts the valid rows on the host
    either way (``row_index``: that count's index, ``None`` when every row is valid or the call is uncompacted).

    Kept (``keep`` true): the valid rows (every row of an uncompacted call) fit one chunk (``0 < T_v <= chunk_size``)
    and the fast backward would run no zero-gradient row skip on them (:func:`.cake_backend._zero_dlogp_skip`: fewer
    valid rows than the device's floor, :func:`.cake_backend.zero_dlogp_min_rows`, or the skip off).  The kept
    backward has no skip: with valid rows of a zero ``dlogp`` it would sum ``dW`` over the rows in the forward's order
    where the skip sums the live rows first, so a call the skip can take keeps the default form, and every output is
    the bits of ``memory_mode="fast"`` -- also in a checkpointed step, which keeps the default form on every pass."""
    if not (
        _ONE_CHUNK_KEEP["on"]
        and backend == "cake"
        and isinstance(X, torch.Tensor)
        and isinstance(W, torch.Tensor)
        and isinstance(labels, torch.Tensor)
        and X.is_cuda
        and torch.is_grad_enabled()
        and (X.requires_grad or W.requires_grad)
        and _main._TOP_SAVED_HOOKS(False) is None
        and torch._C._current_graph_task_id() == -1
        and labels.ndim == 1
        and isinstance(chunk_size, int)
        and not isinstance(chunk_size, bool)
        and chunk_size > 0
    ):
        return None
    T, C = int(labels.shape[0]), int(chunk_size)
    if T == 0 or -(-T // C) >= _main.HIDDEN_COUNT_MIN_CHUNKS:
        return None
    device = X.device
    dz_arm = _resolve_logprob_dz(logprob_dz)
    need_dx = bool(X.requires_grad)
    fused_dx = bool(need_dx and _resolve_dx_finalize(dx_finalize))

    def keep(row_index: Optional[torch.Tensor]) -> bool:
        rows = T if row_index is None else int(row_index.numel())
        return 0 < rows <= C and not _main._zero_dlogp_skip(
            device, row_index, "cake", dz_arm, need_dx, fused_dx
        )

    return keep


class CakeLMHeadDetLogprobOneChunkFunction(CakeLMHeadDetLogprobAutoFunction):
    """A default log-probability call that :func:`_one_chunk_keep_rule` admits: the forward counts its valid rows
    once (:func:`.cake_backend.forward_logprob`'s host count) and right after the count applies the rule's ``keep`` to
    that index -- kept: :func:`.cake_backend.forward_logprob_keep`'s binding keeps the BF16 logits of the valid rows for
    the backward, which forms ``dz`` from them (:class:`.cake_backend.CakeLMHeadDetLogprobKeepFunction`'s backward);
    else the default call's forward and backward (:class:`CakeLMHeadDetLogprobAutoFunction`'s where ``auto``, the
    automatic memory form, reaches the call, else :class:`.cake_backend.CakeLMHeadDetLogprobFunction`'s).  Either way
    the outputs are the bits of ``memory_mode="fast"``, and the work between the count and the first launch is that of
    the default call.  The rule admits no call under saved-tensor hooks, so no checkpoint hand-over applies here."""

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
        keep,
        auto,
    ):
        # every switch resolved once, as in the backend's log-probability function
        compact = _resolve_compact(compact_rows)
        fuse = _resolve_fuse(fuse_dw_cast)
        fused_dx = _resolve_dx_finalize(dx_finalize)
        stream_mode = dw_stream_mode(dw_stream)
        hidden = _resolve_hidden_count(hidden_count)
        tuning = _resolve_gemm_tuning(gemm_tuning)
        dz_arm = _resolve_logprob_dz(logprob_dz)
        lbl_arm = _resolve_logprob_lbl(logprob_lbl)
        ctx.set_materialize_grads(False)
        result = _main.forward_logprob(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            backend=backend,
            compact_rows=compact,
            hidden_count=hidden,
            gemm_tuning=tuning,
            logprob_lbl=lbl_arm,
            keep=keep,
        )
        kept = getattr(result, "logits", None)
        # the kept logits are saved tensors too (see the backend's kept-logits function)
        ctx.save_for_backward(X, W, labels, result.lse, result.row_index, *(kept or ()))
        ctx.one_chunk = ctx.kept = kept is not None
        ctx.auto = bool(auto)
        ctx.chunk_size, ctx.backend, ctx.compact = chunk_size, backend, compact
        ctx.fuse_dw_cast = fuse
        ctx.dx_finalize = fused_dx
        ctx.dw_stream = stream_mode
        ctx.hidden_count = hidden
        ctx.gemm_tuning = tuning
        ctx.logprob_dz = dz_arm
        return result.logp

    @staticmethod
    def backward(ctx, dlogp):
        if ctx.one_chunk:
            grads = _main.CakeLMHeadDetLogprobKeepFunction.backward(ctx, dlogp)
        elif ctx.auto:
            grads = CakeLMHeadDetLogprobAutoFunction.backward(ctx, dlogp)
        else:
            grads = _main.CakeLMHeadDetLogprobFunction.backward(ctx, dlogp)
        return tuple(grads[:2]) + (None,) * 13


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
    memory_mode: Optional[str] = None,
    num_valid_rows=None,
    check_num_valid_rows: bool = False,
) -> torch.Tensor:
    """Differentiable FP32 ``logp [T]`` (0 on ignored rows) for arbitrary downstream losses.  The switches
    ``compact_rows`` / ``fuse_dw_cast`` / ``dx_finalize`` / ``dw_stream`` / ``hidden_count`` / ``gemm_tuning`` and
    their defaults (``None`` = the documented default) as in :func:`cake_lm_head_det_train_loss`, plus the logits GEMM arms
    of this entry: ``logprob_lbl`` (default ``True``, :func:`logprob_lbl_default`) -- the forward's logits GEMM
    writes the row statistics and the label column's logit only -- and ``logprob_dz`` (default ``True``,
    :func:`logprob_dz_default`) -- the backward's recompute writes ``dz`` from its epilogue, no ``row_grad`` launch
    (both bitwise the same outputs).  Every switch is resolved once in the forward; the backward reuses the
    forward's valid-row index (no count, no host synchronization) and recomputes under the same values.
    ``memory_mode`` (``"budget"``: the backward's weight-gradient form of :func:`memory_plan`; the default ``None``,
    the automatic form: the fast backward where it fits, else a budget form -- without ``num_valid_rows``
    :class:`CakeLMHeadDetLogprobAutoFunction`, :mod:`.cake_backend`'s forward and, where it fits, backward),
    ``num_valid_rows`` and ``check_num_valid_rows`` (the forward's count) as in :func:`cake_lm_head_det_train_loss`;
    without ``num_valid_rows``, with ``"fast"`` or a default call whose step forms no BF16 weight gradient on the
    ``cake`` backend, the call is :mod:`.cake_backend`'s ``cake_lm_head_det_train_logprob``, and so is a default call
    under gradient mode off whose ``W`` requires a gradient, with its hidden valid-row count checked against the
    step's budget (:func:`_auto_hidden_count`).  A default call whose valid rows fit one chunk and on which the fast
    backward runs no zero-gradient row skip keeps their BF16 logits from the forward to the backward
    (:func:`_one_chunk_keep_rule`, the one rule; :class:`CakeLMHeadDetLogprobOneChunkFunction`;
    :func:`set_logprob_one_chunk_keep`; bitwise the outputs of ``memory_mode="fast"``, which turns it off for the
    call).  ``memory_mode="keep_logits"`` (:data:`KEEP_LOGITS`;
    ``cake`` backend): :mod:`.cake_backend`'s ``cake_lm_head_det_train_logprob_keep`` -- where a backward can follow, the
    forward keeps the BF16 logits of the valid rows (``2 * T_v * V`` bytes until the backward) and the backward forms
    ``dz`` from them instead of recomputing them (bitwise the same ``logp`` / ``dX`` / ``dW``); where none can, the
    default entry.  With ``num_valid_rows`` (:func:`_logprob_keep_given`) the kept forward runs on the valid-row
    index formed on the device from the caller's count, without a readback (bitwise the same outputs).  Like
    ``"budget"`` it is a ``ValueError`` on the reference backend (the backend-level
    ``cake_lm_head_det_train_logprob_keep`` runs the default entry there)."""
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_CHECK_WITHOUT_COUNT)
    mode: Optional[str] = None
    try:
        mode = _resolve_memory_mode(memory_mode)
    except ValueError:
        # the resolver rejects every mode outside MEMORY_MODES (and the default's); this entry takes KEEP_LOGITS below
        if not (isinstance(memory_mode, str) and memory_mode == KEEP_LOGITS):
            raise
    keep = (
        _one_chunk_keep_rule(
            X, W, labels, chunk_size, backend, compact_rows, dx_finalize, logprob_dz
        )
        if num_valid_rows is None and mode == MEMORY_AUTO
        else None
    )
    if keep is not None:
        # the one-chunk kept logits (:func:`_one_chunk_keep_rule`): the forward picks the form on its own count
        auto = _auto_reaches(backend, X, W)
        hidden = (
            _auto_hidden_count(
                hidden_count,
                X,
                W,
                chunk_size,
                need_dx=bool(X.requires_grad),
                entry="logprob",
            )
            if auto
            else hidden_count
        )
        return CakeLMHeadDetLogprobOneChunkFunction.apply(
            X,
            W,
            labels,
            int(chunk_size),
            backend,
            compact_rows,
            fuse_dw_cast,
            dx_finalize,
            dw_stream,
            hidden,
            gemm_tuning,
            logprob_dz,
            logprob_lbl,
            keep,
            auto,
        )
    if num_valid_rows is None and (
        mode == "fast" or (mode == MEMORY_AUTO and not _auto_reaches(backend, X, W))
    ):
        return _main.cake_lm_head_det_train_logprob(
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
    if memory_mode == KEEP_LOGITS:  # past the fast form: the opt-in kept logits
        if backend != "cake":
            raise ValueError(f"memory_mode={KEEP_LOGITS!r} runs on the cake backend")
        if num_valid_rows is not None:
            return _logprob_keep_given(
                X,
                W,
                labels,
                chunk_size=chunk_size,
                compact_rows=compact_rows,
                fuse_dw_cast=fuse_dw_cast,
                dx_finalize=dx_finalize,
                dw_stream=dw_stream,
                hidden_count=hidden_count,
                gemm_tuning=gemm_tuning,
                logprob_dz=logprob_dz,
                logprob_lbl=logprob_lbl,
                num_valid_rows=num_valid_rows,
                check_num_valid_rows=check_num_valid_rows,
            )
        return _main.cake_lm_head_det_train_logprob_keep(
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
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if num_valid_rows is None and mode == MEMORY_AUTO:
        # a default call whose step forms a BF16 weight gradient: its hidden valid-row count checked against the
        # step's budget (:func:`_auto_hidden_count`); with gradient mode on the default entry's forward and a backward
        # that picks its form, with it off (e.g. a reentrant checkpoint's first pass) the default entry
        hidden = _auto_hidden_count(
            hidden_count,
            X,
            W,
            chunk_size,
            need_dx=bool(X.requires_grad),
            entry="logprob",
        )
        if not torch.is_grad_enabled():
            return _main.cake_lm_head_det_train_logprob(
                X,
                W,
                labels,
                chunk_size=chunk_size,
                backend=backend,
                compact_rows=compact_rows,
                fuse_dw_cast=fuse_dw_cast,
                dx_finalize=dx_finalize,
                dw_stream=dw_stream,
                hidden_count=hidden,
                gemm_tuning=gemm_tuning,
                logprob_dz=logprob_dz,
                logprob_lbl=logprob_lbl,
            )
        return CakeLMHeadDetLogprobAutoFunction.apply(
            X,
            W,
            labels,
            int(chunk_size),
            backend,
            compact_rows,
            fuse_dw_cast,
            dx_finalize,
            dw_stream,
            hidden,
            gemm_tuning,
            logprob_dz,
            logprob_lbl,
        )
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
        mode,
        num_valid_rows,
        check_num_valid_rows,
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
    memory_mode: Optional[str] = None,
    num_valid_rows=None,
    check_num_valid_rows: bool = False,
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
    ``memory_mode`` (``"budget"``: the backward's weight-gradient form of :func:`memory_plan`; the default ``None``,
    the automatic form: as in :func:`cake_lm_head_det_train_logprob`, with
    :class:`CakeLMHeadDetLogprobTemperatureAutoFunction` for a call with a temperature), ``num_valid_rows`` and
    ``check_num_valid_rows`` (the forward's count) as in :func:`cake_lm_head_det_train_loss`; without
    ``num_valid_rows``, with ``"fast"`` or a default call whose step forms no BF16 weight gradient on the ``cake``
    backend, the call is :mod:`.cake_backend`'s ``cake_lm_head_det_train_logprob_temperature`` (under gradient mode
    off with a ``W`` that requires a gradient too, its hidden valid-row count checked as in
    :func:`cake_lm_head_det_train_logprob`).

    ``temperature`` (default ``None``, the same as ``1.0``: no temperature):
    a positive number or a float32 ``[T]`` tensor on ``X``'s device (one value per
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
    ``logprob_dz`` must stay on (the temperature forms are those arms).  ``memory_mode="keep_logits"`` takes no
    temperature (``ValueError``: the kept logits' gradient form ``row_grad`` has no temperature form); with a scalar
    whose FP32 value is 1.0 it is the call without a temperature."""
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_CHECK_WITHOUT_COUNT)
    if isinstance(memory_mode, str) and memory_mode == KEEP_LOGITS:
        # the kept logits form dz with row_grad, which has no temperature form; a scalar whose FP32 value is 1.0
        # resolves to no temperature and runs the kept form as the call without one
        if (
            isinstance(temperature, torch.Tensor) and temperature.ndim == 1
        ) or _resolve_temperature(temperature, 0, torch.device("cpu")) is not None:
            raise ValueError(
                f"memory_mode={KEEP_LOGITS!r} takes no temperature (the kept logits' gradient form has no temperature "
                "form): pass temperature=None, or memory_mode=None, 'fast' or 'budget'"
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
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
        )
    mode = _resolve_memory_mode(memory_mode)
    if num_valid_rows is None and (
        mode == "fast" or (mode == MEMORY_AUTO and not _auto_reaches(backend, X, W))
    ):
        return _main.cake_lm_head_det_train_logprob_temperature(
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
            temperature=temperature,
            check_temperature=check_temperature,
        )
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    plain = temperature is None or (
        type(temperature) is float and temperature == 1.0
    )  # the default path: no temperature operation (a scalar 1.0 resolves to none)
    if num_valid_rows is None and mode == MEMORY_AUTO:
        # a default call whose step forms a BF16 weight gradient (as in :func:`cake_lm_head_det_train_logprob`): with
        # gradient mode on the default entry's forward and a backward that picks its form, with it off the default
        # entry, its hidden valid-row count checked against the step's budget either way
        hidden = _auto_hidden_count(
            hidden_count,
            X,
            W,
            chunk_size,
            need_dx=bool(X.requires_grad),
            entry="logprob",
        )
        if not torch.is_grad_enabled():
            return _main.cake_lm_head_det_train_logprob_temperature(
                X,
                W,
                labels,
                chunk_size=chunk_size,
                backend=backend,
                compact_rows=compact_rows,
                fuse_dw_cast=fuse_dw_cast,
                dx_finalize=dx_finalize,
                dw_stream=dw_stream,
                hidden_count=hidden,
                gemm_tuning=gemm_tuning,
                logprob_dz=logprob_dz,
                logprob_lbl=logprob_lbl,
                temperature=temperature,
                check_temperature=check_temperature,
            )
        if plain:
            return CakeLMHeadDetLogprobAutoFunction.apply(
                X,
                W,
                labels,
                int(chunk_size),
                backend,
                compact_rows,
                fuse_dw_cast,
                dx_finalize,
                dw_stream,
                hidden,
                gemm_tuning,
                logprob_dz,
                logprob_lbl,
            )
        return CakeLMHeadDetLogprobTemperatureAutoFunction.apply(
            X,
            W,
            labels,
            int(chunk_size),
            backend,
            compact_rows,
            fuse_dw_cast,
            dx_finalize,
            dw_stream,
            hidden,
            gemm_tuning,
            logprob_dz,
            logprob_lbl,
            temperature,
            check_temperature,
        )
    if plain:
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
            mode,
            num_valid_rows,
            check_num_valid_rows,
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
        mode,
        num_valid_rows,
        check_num_valid_rows,
        temperature,
        check_temperature,
    )


PRELOAD_ROWS = (
    1 << 16
)  # rows of the int32 count constant (1, 2, ...) preload allocates; longer calls extend it


def cake_lm_head_det_train_preload(
    device: Optional[Union[torch.device, int, str]] = None,
    *,
    temperature: bool = False,
) -> None:
    """Do part of the one-time work of a process's first call ahead of it, then wait for the device once.

    ``cake_lm_head_det_train_preload`` loads every library of the stages ``device``'s architecture registers in ``cake_jit.STAGES`` (a JIT
    build when the package ships no prebuilt ``aot/`` library; the candidate-support stages of ``cake_support`` are
    not among them and still load at their first call), creates the calling thread's side stream and its fork event,
    the hidden count's pinned cell and event and the int32 count constant (another thread creates its own at its
    first call) and the constant cells, launches each PyTorch operator of the valid-row count paths once on a few
    rows (``ge``, ``cumsum``, ``searchsorted``, ``eq``, ``_assert_async``, ``nonzero``, ``index_select``,
    ``index_copy_``, ``where``) and then synchronizes ``device``.  It does not launch the package's kernels: under the
    driver's lazy module loading (``CUDA_MODULE_LOADING=LAZY``, the default) each kernel form still loads at its first
    launch.  ``device``: a CUDA device, its index or ``None`` (the current device).

    ``temperature`` (default ``False``): the temperature forms of the log-probability entry
    (:data:`TEMPERATURE_STAGES`) are loaded only with ``temperature=True`` -- callers that never pass a temperature
    load (or JIT-build) no temperature library -- which also launches the temperature path's PyTorch operators once
    (``gt``, ``lt``, ``&``, ``all``, ``full``, ``clamp_max``, ``index_select``)."""
    if device is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        dev = torch.device("cuda", device)
    else:
        dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(
            f"cake_lm_head_det_train_preload needs a CUDA device, got {dev}"
        )
    index = int(dev.index if dev.index is not None else torch.cuda.current_device())
    dev = torch.device("cuda", index)
    module_name, record = record_for(dev)
    record_abi(record)
    stages = (
        registered_stages_with_temperature(module_name)
        if temperature
        else registered_stages(module_name)
    )
    # the temperature forms load with cake_lm_head_det_train_preload(temperature=True) or at a call's first use
    for stage in stages:
        load_cake_lm_head_det_loss_module(module_name, stage)
    with torch.cuda.device(index):
        _dw_stream(index)
        _hidden_count_cell(index)
        _device_constants(index)
        _hidden_count_arange(index, PRELOAD_ROWS)
        labels = torch.tensor(
            [0, IGNORE_INDEX, 3, 1, IGNORE_INDEX, 2, 0, 1],
            dtype=torch.int64,
            device=dev,
        )
        valid, pos, idx_full = compaction_index(
            labels, ar=_hidden_count_arange(index, int(labels.numel()))
        )
        _assert_on_device(pos[-1] == int(pos.numel() - 2), "preload")
        index_rows = valid.nonzero().squeeze(1)
        rows = torch.zeros((8, 8), dtype=torch.bfloat16, device=dev)
        picked = torch.index_select(rows, 0, index_rows)
        torch.zeros_like(rows).index_copy_(0, index_rows, picked)
        torch.index_select(labels, 0, idx_full[: int(index_rows.numel())])
        torch.where(labels < 0, labels, torch.where(labels >= 1, labels - 1, 1 << 30))
        if temperature:
            # the temperature path's operators: the value check, the scalar fill and the gathers
            tau = torch.full((8,), 0.5, dtype=torch.float32, device=dev)
            _assert_on_device(((tau > 0) & (tau < math.inf)).all(), "preload")
            torch.index_select(tau, 0, index_rows)
            torch.index_select(
                tau,
                0,
                idx_full.clamp_max(int(tau.numel()) - 1),
                out=torch.empty_like(tau),
            )
        torch.cuda.synchronize(dev)
