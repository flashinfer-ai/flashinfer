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
"""

import functools
import inspect
import json
import os
import struct
import warnings
from types import ModuleType
from typing import Any, Dict, NoReturn, Optional, Union

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry points; validation of the backend contract, the
# chunk plans, the workspace, launch binding, autograd wrappers and JIT
# registration live in flashinfer.experimental.cake_lm_head_det_loss (the
# deterministic package, entries ``chunked_lm_head_det_*``) and
# flashinfer.experimental.cake.cake_lm_head_loss (the non-deterministic
# package).  The det-free entries ``chunked_lm_head_{loss,logprob,preload,
# support_logprob,support_kl}`` take ``deterministic: bool = True`` and
# dispatch (see the section above them at the end of this module).
#
# The package backend takes every switch as a keyword argument and reads no
# environment variable.  The FLASHINFER_CAKE_LM_HEAD_LOSS_* variables below
# are this wrapper's A/B knobs: a switch keyword left at ``None`` falls back
# to its variable here, and the resolved value is passed down.  The two
# binding-cache variables are read once, when this module is imported, and
# applied to the backend's cache on first use (the first call of either
# entry point).

_FEATURE = "chunked LM-head + loss training kernels (SM100/SM103/SM107)"

_CHECK_WITHOUT_COUNT = (
    "check_num_valid_rows=True needs num_valid_rows (the count it checks)"
)

# "0" turns valid-row compaction off (``compact_rows``); anything else: on.
COMPACT_ROWS_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS"
# "0" turns the fused weight-gradient cast off (``fuse_dw_cast``).
FUSE_DW_CAST_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST"
# "0" turns the fused dX finalize off (``dx_finalize``).
DX_FINALIZE_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE"
# "0" / "false" / "off" / "no": never; "1" / "true" / "on" / "yes": every
# multi-chunk call; anything else (the default): "auto" (``dw_stream``).
DW_STREAM_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM"
# "0" turns the hidden valid-row count off (``hidden_count``).
HIDDEN_COUNT_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT"
# A JSON object of explicit per-GEMM knobs (``gemm_tuning``); unset / empty:
# none.
GEMM_TUNING_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING"
# "0" disables the backend's binding cache (read at import, applied on first
# use).
BINDING_CACHE_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE"
# Positive integer: the binding cache's capacity (read at import, applied on
# first use; unset / blank: the backend's default of 64 bindings).
BINDING_CACHE_CAPACITY_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY"

_DW_STREAM_OFF = ("0", "false", "off", "no")
_DW_STREAM_ON = ("1", "true", "on", "yes")


def _env_switch(value: Optional[bool], name: str) -> bool:
    """``value`` when given, else the variable ``name``: on unless set to ``"0"``."""
    if value is not None:
        return bool(value)
    return os.environ.get(name, "1") != "0"


def _env_dw_stream(value: Optional[Union[bool, str]]) -> Union[bool, str]:
    """``value`` when given, else ``FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM`` as
    ``False`` (never), ``True`` (every multi-chunk call) or ``"auto"``."""
    if value is not None:
        return value
    raw = os.environ.get(DW_STREAM_ENV, "auto").strip().lower()
    if raw in _DW_STREAM_OFF:
        return False
    if raw in _DW_STREAM_ON:
        return True
    return "auto"


def _env_gemm_tuning(
    value: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """``value`` when given, else ``FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING`` parsed
    as a JSON object (``None`` when unset / blank; invalid JSON or a non-object is a
    ``ValueError``); the backend validates the knobs themselves."""
    if value is not None:
        return value
    raw = os.environ.get(GEMM_TUNING_ENV, "").strip()
    if not raw:
        return None
    try:
        knobs = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"${GEMM_TUNING_ENV} must be a JSON object of GEMM knobs: {exc}"
        ) from exc
    if not isinstance(knobs, dict):
        raise ValueError(
            f"${GEMM_TUNING_ENV} must be a JSON object of GEMM knobs, got {raw!r}"
        )
    return knobs


def _resolve_knobs(
    *,
    compact_rows: Optional[bool],
    fuse_dw_cast: Optional[bool],
    dx_finalize: Optional[bool],
    dw_stream: Optional[Union[bool, str]],
    hidden_count: Optional[bool],
    gemm_tuning: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """The switch keywords one call passes to the backend: every explicit value
    as given, every ``None`` replaced by its environment variable."""
    return dict(
        compact_rows=_env_switch(compact_rows, COMPACT_ROWS_ENV),
        fuse_dw_cast=_env_switch(fuse_dw_cast, FUSE_DW_CAST_ENV),
        dx_finalize=_env_switch(dx_finalize, DX_FINALIZE_ENV),
        dw_stream=_env_dw_stream(dw_stream),
        hidden_count=_env_switch(hidden_count, HIDDEN_COUNT_ENV),
        gemm_tuning=_env_gemm_tuning(gemm_tuning),
    )


def _binding_cache_settings() -> Dict[str, Any]:
    """``enabled`` / ``capacity`` of the backend's binding cache from the two
    binding-cache variables (``capacity`` ``None`` = the backend's default)."""
    enabled = os.environ.get(BINDING_CACHE_ENV, "1") != "0"
    raw = os.environ.get(BINDING_CACHE_CAPACITY_ENV)
    capacity: Optional[int] = None
    if raw is not None and raw.strip():
        try:
            capacity = int(raw)
        except ValueError:
            capacity = 0
        if capacity < 1:
            raise ValueError(
                f"{BINDING_CACHE_CAPACITY_ENV} must be a positive integer, got {raw!r}"
            )
    return dict(enabled=enabled, capacity=capacity)


def _configure_binding_cache(
    cake_backend: ModuleType, settings: Dict[str, Any]
) -> None:
    """Apply ``settings`` (``enabled`` / ``capacity`` as returned by
    :func:`_binding_cache_settings`) to ``cake_backend.BINDING_CACHE`` in place;
    a ``capacity`` of ``None`` is the backend's default."""
    capacity = settings["capacity"]
    cake_backend.BINDING_CACHE.configure(
        enabled=settings["enabled"],
        capacity=cake_backend.BINDING_CACHE_DEFAULT_CAPACITY
        if capacity is None
        else capacity,
    )


@functools.cache
def _cake_backend() -> ModuleType:
    # The backend is imported on first use (experimental containment: core
    # reaches kernels and their dependencies through a deferred import) and
    # the binding-cache settings read at import are applied to it once.
    from .experimental.cake_lm_head_det_loss import cake_backend

    _configure_binding_cache(cake_backend, _BINDING_CACHE_SETTINGS)
    return cake_backend


@functools.cache
def _cake_backend_opt_in() -> ModuleType:
    # The opt-in forms (``memory_mode`` / ``num_valid_rows``), imported on
    # their first use; a call with neither never imports them.  Their own
    # binding cache takes the same settings.
    _cake_backend()
    from .experimental.cake_lm_head_det_loss import cake_backend_opt_in

    _configure_binding_cache(cake_backend_opt_in, _BINDING_CACHE_SETTINGS)
    return cake_backend_opt_in


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_det_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div: Optional[float] = None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
    return_logp: bool = False,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[Dict[str, Dict[str, Any]]] = None,
    memory_mode: Optional[str] = None,
    num_valid_rows: Optional[int] = None,
    check_num_valid_rows: bool = False,
):
    r"""Differentiable chunked LM-head projection + loss with a memory-bounded
    backward, SM100 / SM103 / SM107 only.

    Computes the projection ``z = X @ W^T`` over token chunks of ``chunk_size``
    rows, the per-row log-probability of the selected token
    ``logp_t = z[t, y_t] - logsumexp_v z[t, v]`` and one of two objectives,
    without ever materializing a vocabulary-sized tensor of more than
    ``chunk_size`` rows.  The forward already produces the FP32 gradient
    accumulators of the trainable inputs (three GEMMs per chunk, no logits
    recomputation); the backward applies the incoming scalar gradient and
    performs the single cast to the output dtypes.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, H]`` hidden states, row-major with any leading stride
        (``H`` a multiple of 256; a stride whose byte pitch is not a multiple
        of 16 is copied to a contiguous tensor first).
    W : torch.Tensor
        BF16 ``[V, H]`` output weight, contiguous (``V`` a multiple of 256).
    labels : torch.Tensor
        int64 ``[T]``; ``-100`` marks an ignored row (zero loss, zero gradient,
        ``logp = 0``).
    objective : str
        ``"ce"``: ``loss = -sum(logp[valid]) / loss_div``.
        ``"policy"``: ``loss = -sum(loss_weights * min(exp(logp - infer_logp), 2))``
        over valid rows; the per-row logit-gradient scale is
        ``-loss_weights_t * ratio_t`` when ``ratio_t <= 2`` and ``0`` above the
        clipping boundary.
    loss_div : Optional[float]
        Positive scalar divisor of the cross-entropy objective (caller-supplied,
        never replaced by a local token mean).  Required for ``"ce"``.
    infer_logp : Optional[torch.Tensor]
        FP32 ``[T]`` inference log-probabilities of the policy objective.
    loss_weights : Optional[torch.Tensor]
        FP32 ``[T]`` signed weights of the policy objective; they already
        include masking and normalization (ignored rows carry weight 0).
    chunk_size : int
        Token chunk ``C`` (default 4096).  Every logits / dlogits / probability
        buffer spans at most ``C`` tokens; a batch smaller than ``C`` is one
        chunk, and a tail ``T % C`` is neither dropped nor padded.  Changing
        ``C`` may change reduction rounding, never the loss definition.
    return_logp : bool
        Also return the detached FP32 ``logp [T]`` (a diagnostic output).
    grad_weight_dtype : torch.dtype
        dtype of the weight gradient: ``torch.bfloat16`` (default) or
        ``torch.float32``.  ``dW`` is accumulated in FP32 across chunks and
        cast once at the output boundary.  Through this autograd entry the
        value must equal ``W.dtype`` (the autograd engine casts every gradient
        to its leaf's dtype); an FP32 ``dW`` for a BF16 ``W`` is available from
        the explicit ``cake_backend.forward_loss`` / ``backward_loss`` pair.
    deterministic : bool
        Only ``True`` is available: fixed sequential chunk order, no atomics,
        bitwise reproducible ``loss``, ``logp``, ``dX`` and ``dW``.
    backend : str
        Only ``"cake"``.
    compact_rows : bool, optional
        Chunk over the valid rows only (rows whose label is not ``-100``):
        the valid-row index is formed once per call, each chunk's rows of
        ``X`` are gathered into one reusable ``[chunk_size, H]`` buffer and
        ``logp`` / ``dX`` are scattered back with exact zeros on the ignored
        rows -- the same computation over fewer rows (per-row ``logp`` is
        bitwise that of the uncompacted path; the ``loss`` / ``dW`` reductions
        run over different chunk boundaries and differ by FP32 rounding).
        ``None`` (default) reads ``FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS``
        (on unless set to ``0``; the backend's own default is ``True``).
        Compaction costs one device synchronization per call, the valid-row
        count (the log-probability backward reuses the forward's index).
    fuse_dw_cast : bool, optional
        Run the last chunk's weight-gradient GEMM in the backward, where the
        upstream scalar gradient is known, with the scale and the output cast
        fused into its epilogue: the forward accumulates the chunks before it
        in FP32 (no accumulator at all for a one-chunk call), the backward
        writes ``dW = cast(g * (dW_acc + dz_c^T @ X_c))`` straight from the
        GEMM and the separate pass over the ``[V, H]`` accumulator disappears.
        The same FP32 operations in the same order: ``dW`` is bitwise the
        unfused result.  The last chunk's BF16 ``dlogits`` rows and a view of
        its rows of ``X`` are saved for the backward (an in-place write to
        ``X`` between the forward and the backward raises PyTorch's
        saved-tensor version error).  ``None`` (default) reads
        ``FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST`` (on unless set to ``0``;
        the backend's own default is ``True``).
    dx_finalize : bool, optional
        Add the K-slice slabs of a sliced ``dX`` GEMM with one fixed-order
        kernel and, with ``compact_rows``, write the ``[T, H]`` ``dX`` in one
        pass (``bf16(g * dX_acc)`` on the valid rows, exact zeros elsewhere)
        instead of casting, zero-filling and scattering the compact rows; the
        same FP32 operations in the same order, so ``dX`` is bitwise the same.
        ``None`` (default) reads ``FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE``
        (on unless set to ``0``; the backend's own default is ``True``).
    dw_stream : bool or str, optional
        Launch each chunk's weight-gradient accumulate GEMM on a per-device
        side stream, forked after the chunk's row gradients and joined before
        the chunk buffer is reused and before the call returns, so it overlaps
        the tail of the chunk's ``dX`` GEMM: ``"auto"`` for calls of three or
        more chunks, ``True`` for every multi-chunk call, ``False`` never; the
        same kernels in the same order per kernel, so every output is bitwise
        the same.  ``None`` (default) reads
        ``FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM`` (``0`` / ``false`` / ``off``
        / ``no`` -> ``False``, ``1`` / ``true`` / ``on`` / ``yes`` -> ``True``,
        anything else -> ``"auto"``; the backend's own default is ``"auto"``).
    hidden_count : bool, optional
        Form a compacted call's valid-row index on the device and queue chunk
        0's row gather and logits GEMM (its device-count form, bounded by the
        count read from device memory) before the count reaches the host
        through a pinned cell and a CUDA event, instead of counting and
        indexing on the host before the first launch; the same kernels per
        row, so every output is bitwise the same.  ``None`` (default) reads
        ``FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT`` (on unless set to ``0``;
        the backend's own default is ``True``).
    gemm_tuning : dict, optional
        Explicit per-GEMM knobs, ``{"logits": {...}, "dx": {...}, "dw": {...}}``
        (e.g. ``{"dx": {"tile_n": 256, "k_slices": 2}}``; a knob given as
        ``None`` pins the default form), winning over the per-chunk rules for
        the knobs they name; a form the compiled program does not register
        fails closed.  ``None`` (default) reads
        ``FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING`` as a JSON object (unset or
        blank: no explicit knob, the backend's own default; invalid JSON is a
        ``ValueError``).
    memory_mode : str, optional
        ``None`` (the default) or ``"fast"``: the fastest form; an explicit
        ``"fast"`` first goes through the opt-in dispatch
        (``cake_backend_opt_in``) and then runs the same entry as ``None``.
        ``"budget"``: bound the step peak (forward through backward, outputs
        included) by the bytes ``B`` of a head that materializes the BF16
        logits of the valid rows (``B = 2 * T_v * V + 2 * V * H + 2 * T * H``);
        the planner targets ``B`` less a 2 % margin: per call, the fastest
        exact weight-gradient form whose predicted peak fits -- the fused or
        the unfused cast, or the forward accumulating the leading vocabulary
        rows and the backward recomputing the others in vocabulary blocks
        (``cake_backend_opt_in.memory_plan``).  Bitwise the same ``loss`` /
        ``logp`` / ``dX`` / ``dW`` as ``"fast"``; the recompute forms run more
        GEMM work.  The bound assumes a backward that frees the graph
        (``retain_graph=True`` keeps the forward's state); its margin is
        measured on GB200, GB300 and B200 (see the package README).
        ``"keep_logits"`` belongs to ``chunked_lm_head_det_logprob`` (a
        ``ValueError`` here).  No environment variable.
    num_valid_rows : int, optional
        The number of labels ``>= 0`` (not ``-100``), as a host integer (a
        Python or NumPy integer or a CPU scalar tensor; a CUDA tensor is
        rejected).  With compaction the forward then plans on it without
        waiting for queued device work: the valid-row index is formed on the
        device and the count is checked there by one asynchronous assertion
        -- a wrong count ends the CUDA context with a device-side assertion.
        Exception: an ``sm_103a`` call whose library CUDA 13.0 built keeps the
        host-count schedule and reads the count back once (see the package
        README's toolchain note).  ``None`` (default): the forward counts the
        valid rows itself, which waits for the device once per call.  Not
        derived from ``loss_div``.
    check_num_valid_rows : bool
        Read the count back and raise ``ValueError`` when ``num_valid_rows``
        is wrong (one synchronization; for debugging).  ``True`` without
        ``num_valid_rows`` is a ``ValueError``.

    Notes
    -----
    Precision boundary: BF16 GEMM output promoted to FP32 for the max /
    log-sum-exp / loss arithmetic, BF16 ``dlogits``, FP32 ``dW`` accumulation
    with one cast at the output, BF16 ``dX`` (FP32 accumulation, one cast).
    Gradients are produced only for inputs that require them (a frozen ``X``
    or ``W`` skips its GEMM).  The saved accumulators are re-scaled, never
    mutated, so a retained graph may run the backward repeatedly.  ``T == 0``
    gives loss 0, an empty ``logp`` and zero gradients.  Under the backend's
    default gradient placement (``cake_backend.set_gradient_placement``,
    ``"auto"``) a call no backward can follow (grad mode off, also the first
    pass of a reentrant ``torch.utils.checkpoint``) computes the loss only,
    and inside a non-reentrant ``torch.utils.checkpoint`` the recomputation
    forms the gradient accumulators instead of the checkpointed forward (one
    gradient pass per step); every output is bitwise the same.

    A call with neither ``memory_mode`` nor ``num_valid_rows`` runs the
    backend's default entry (``cake_backend``) after this function's argument
    checks (``check_num_valid_rows``, then ``memory_mode`` and
    ``num_valid_rows`` against ``None``).  A call with either goes through
    ``cake_backend_opt_in`` (bitwise the same outputs): an explicit
    ``memory_mode="fast"`` without a count then runs the same default entry,
    every other such call takes its opt-in forms; with
    ``memory_mode="budget"`` a backward that frees the graph releases the
    saved accumulators as soon as it has used them (a second backward then
    raises).

    ``chunked_lm_head_det_preload`` loads the libraries and creates the
    per-device state ahead of a process's first call.

    Host side: inputs are validated and bound on the first call for an input
    binding (``data_ptr``, shape, stride and dtype of every input plus the
    options); later calls with the same binding launch from the remembered
    argument plans
    (``flashinfer.experimental.cake_lm_head_det_loss.cake_backend.BINDING_CACHE``).
    ``FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE=0`` disables that cache and
    ``FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY`` (a positive
    integer; default 64) sets its capacity: both are read when this module is
    imported and applied through ``BINDING_CACHE.configure`` on the first call
    of either entry point.  Peak temporary memory is reported separately
    from the weights, the outputs and the FP32 accumulators by
    ``cake_backend.memory_report``.

    Returns
    -------
    loss : torch.Tensor
        FP32 scalar.
    logp : torch.Tensor
        Detached FP32 ``[T]``, only with ``return_logp=True``.
    """
    if backend != "cake":
        raise ValueError(
            "the chunked LM-head + loss kernels currently support backend='cake'"
        )
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_CHECK_WITHOUT_COUNT)
    if memory_mode is not None or num_valid_rows is not None:
        return _cake_backend_opt_in().chunked_lm_head_det_loss(
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
            deterministic=deterministic,
            backend="cake",
            **_resolve_knobs(
                compact_rows=compact_rows,
                fuse_dw_cast=fuse_dw_cast,
                dx_finalize=dx_finalize,
                dw_stream=dw_stream,
                hidden_count=hidden_count,
                gemm_tuning=gemm_tuning,
            ),
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
        )
    return _cake_backend().chunked_lm_head_det_loss(
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
        deterministic=deterministic,
        backend="cake",
        **_resolve_knobs(
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
        ),
    )


def _chunked_lm_head_det_logprob_temperature(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 4096,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[Dict[str, Dict[str, Any]]] = None,
    memory_mode: Optional[str] = None,
    num_valid_rows: Optional[int] = None,
    check_num_valid_rows: bool = False,
    temperature: Union[None, float, torch.Tensor] = None,
    check_temperature: bool = False,
) -> torch.Tensor:
    """:func:`chunked_lm_head_det_logprob` with a ``temperature`` (not ``None``): the same routing, through the
    backends' ``chunked_lm_head_det_logprob_temperature`` entries."""
    if backend != "cake":
        raise ValueError(
            "the chunked LM-head + loss kernels currently support backend='cake'"
        )
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_CHECK_WITHOUT_COUNT)
    if memory_mode is not None or num_valid_rows is not None:
        return _cake_backend_opt_in().chunked_lm_head_det_logprob_temperature(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            deterministic=deterministic,
            backend="cake",
            **_resolve_knobs(
                compact_rows=compact_rows,
                fuse_dw_cast=fuse_dw_cast,
                dx_finalize=dx_finalize,
                dw_stream=dw_stream,
                hidden_count=hidden_count,
                gemm_tuning=gemm_tuning,
            ),
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
            temperature=temperature,
            check_temperature=check_temperature,
        )
    return _cake_backend().chunked_lm_head_det_logprob_temperature(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        deterministic=deterministic,
        backend="cake",
        **_resolve_knobs(
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
        ),
        temperature=temperature,
        check_temperature=check_temperature,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_det_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 4096,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[Dict[str, Dict[str, Any]]] = None,
    memory_mode: Optional[str] = None,
    num_valid_rows: Optional[int] = None,
    check_num_valid_rows: bool = False,
    temperature: Union[None, float, torch.Tensor] = None,
    check_temperature: bool = False,
) -> torch.Tensor:
    r"""Differentiable chunked selected-token log-probability, SM100 / SM103 / SM107 only.

    Returns the FP32 ``logp [T]`` (``logp_t = z[t, y_t] - logsumexp_v z[t, v]``
    for valid rows, 0 for rows whose label is ``-100``) of the projection
    ``z = X @ W^T`` for arbitrary downstream losses; the per-row gradient
    ``dlogp`` arrives from autograd.  The forward saves the FP32 log-sum-exp
    and the valid-row index; the backward recomputes each chunk's logits (four
    GEMMs per chunk; the recompute writes the logit gradient from its
    epilogue) and produces BF16
    ``dX`` and ``dW`` through FP32 accumulators with one cast each (with
    ``fuse_dw_cast`` the last chunk's weight-gradient GEMM writes the BF16
    ``dW`` from its epilogue).  Arguments as in :func:`chunked_lm_head_det_loss`,
    including the switches ``compact_rows``, ``fuse_dw_cast``, ``dx_finalize``,
    ``dw_stream``, ``hidden_count`` and ``gemm_tuning`` with the same defaults
    (``None`` reads the same ``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` variable; every
    switch is resolved once in the forward and the backward recomputes under
    the same values); ``dW`` is returned in BF16.  ``memory_mode``
    (``"budget"``: the backward's weight-gradient form within the same bound),
    ``num_valid_rows`` and ``check_num_valid_rows`` (the forward's count) as in
    :func:`chunked_lm_head_det_loss`.  ``memory_mode="keep_logits"`` (this
    entry only, without ``num_valid_rows``; opt-in): where a backward can
    follow (grad mode on, ``X`` or ``W`` trainable) the forward keeps the BF16
    logits of the valid rows, ``2 * T_v * V`` bytes from the forward to the
    backward (the chunk bound of the memory rule does not hold for them), and
    the first backward forms the logit gradient from them instead of
    recomputing the logits (three GEMMs per chunk instead of four; a second
    backward through a retained graph recomputes); where none can follow
    (``torch.no_grad``, ``torch.inference_mode``, frozen inputs) the call is
    the default one.  Bitwise the same ``logp``, ``dX`` and ``dW``.

    ``temperature`` (default ``None``, the same as ``1.0``: no temperature)
    is a positive Python number or a float32 ``[T]`` tensor on ``X``'s device
    (one value per row, any stride);
    it is not differentiable (a tensor with ``requires_grad=True`` is
    rejected).  The result is

    ``logp_t = z[t, y_t] / tau_t - logsumexp_v(z[t, v] / tau_t)``

    with the gradient ``(1[v = y_t] - softmax(z_t / tau_t)_v) / tau_t`` per
    logit.  The BF16 projection ``z`` is unchanged; each logit is widened to
    FP32 and divided once (IEEE FP32 division), the upstream ``dlogp_t`` is
    divided once per row, and ``dX`` / ``dW`` see the temperature only
    through the logit gradient.  A scalar ``s`` gives the same bits as
    ``torch.full((T,), s)``; ``None`` and ``1.0`` run the path without a
    temperature.
    Every value must be positive and finite, also on ignored rows (label
    ``-100``).  A bad tensor value stops the CUDA stream with a device-side
    assertion, which ends the CUDA context (no host synchronization);
    ``check_temperature=True`` reads the check back once and raises
    ``ValueError`` instead.  Outputs are defined while
    ``|z / tau_t| < 3e38`` on every row.  ``memory_mode="keep_logits"``
    takes no temperature (``ValueError``).
    """
    if temperature is not None:
        return _chunked_lm_head_det_logprob_temperature(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            deterministic=deterministic,
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
            temperature=temperature,
            check_temperature=check_temperature,
        )
    if backend != "cake":
        raise ValueError(
            "the chunked LM-head + loss kernels currently support backend='cake'"
        )
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_CHECK_WITHOUT_COUNT)
    if memory_mode is not None or num_valid_rows is not None:
        return _cake_backend_opt_in().chunked_lm_head_det_logprob(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            deterministic=deterministic,
            backend="cake",
            **_resolve_knobs(
                compact_rows=compact_rows,
                fuse_dw_cast=fuse_dw_cast,
                dx_finalize=dx_finalize,
                dw_stream=dw_stream,
                hidden_count=hidden_count,
                gemm_tuning=gemm_tuning,
            ),
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
        )
    return _cake_backend().chunked_lm_head_det_logprob(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        deterministic=deterministic,
        backend="cake",
        **_resolve_knobs(
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
        ),
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_det_preload(
    device: Optional[Union[torch.device, int, str]] = None,
    *,
    temperature: bool = False,
):
    r"""Do part of the one-time work of a process's first chunked LM-head call
    ahead of it, then wait for the device once (SM100 / SM103 / SM107 only).

    Loads every generated library of the loss and log-probability entries for
    ``device``'s architecture (a JIT build when the package ships no prebuilt
    library; the candidate-support entries' libraries still load at their first
    call), creates the per-device side stream (PyTorch's stream pool), the
    count cells and constants, launches the PyTorch operators of the valid-row
    count paths once and synchronizes.  It
    does not launch the package's kernels: under the driver's lazy module
    loading (``CUDA_MODULE_LOADING=LAZY``, the default) each kernel form still
    loads at its first launch.  ``device``: a CUDA device, its index or
    ``None`` (the current device).
    ``temperature=True`` also loads the libraries of the
    :func:`chunked_lm_head_det_logprob` temperature path and launches its PyTorch
    operators once; by default they are not loaded (callers that never pass
    a temperature build no temperature library).
    """
    _cake_backend_opt_in().preload(device, temperature=temperature)


@functools.cache
def _cake_support() -> ModuleType:
    # The candidate-support entries live in their own module of the package; it
    # binds its kernels through the backend's cache, configured on first use.
    _cake_backend()
    from .experimental.cake_lm_head_det_loss import cake_support

    return cake_support


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_det_support_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    support_ids: torch.Tensor,
    support_offsets: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    *,
    k_cap: int,
    temperature: Union[None, float, torch.Tensor] = 1.0,
    chunk_size: Optional[int] = None,
    check_inputs: bool = False,
    backend: str = "cake",
) -> torch.Tensor:
    r"""Differentiable log-probabilities over a ragged candidate support per token, SM100 / SM103 / SM107 only.

    The candidates of row ``t`` are ``support_ids[support_offsets[t]:support_offsets[t + 1]]`` (int32 or
    int64 ids, unique inside a row, any order; int32 or int64 offsets, ``support_offsets[0] = 0``,
    nondecreasing, ``support_offsets[T] = M``; at most ``k_cap`` per row, never trimmed).  With
    ``z = X @ W^T`` as a BF16 GEMM (FP32 accumulation, one rounding) and the temperature ``tau_t``:
    ``logp_j = z_j / tau_t - logsumexp_{k in row t}(z_k / tau_t)`` in FP32, ``0`` on a row of one
    candidate or none.  ``labels is None``: the packed FP32 ``logp [M]`` in caller order; otherwise
    the selected FP32 ``logp [T]`` at the int64 ``labels`` (``-100``: 0 with zero gradient; any other
    label must be one of the row's candidates).  ``temperature``: a finite positive number or an FP32
    ``[T]`` tensor on ``X``'s device (not differentiable; checked on every row).  ``chunk_size``: the
    token chunk of the dense pass form and of the large-support weight gradient (default 4096).
    Invalid support is detected on the device: ``check_inputs=False`` queues an asynchronous
    assertion (no synchronization; a violation ends the CUDA context), ``check_inputs=True`` reads the
    validation word back once and raises ``ValueError`` naming the rule and the row.  Gradients flow
    to ``X`` and ``W`` (BF16, FP32 accumulation, one cast each); outputs and gradients are bitwise
    reproducible run to run.  ``X [T, H]`` BF16 row-major (``H`` a multiple of 256), ``W [V, H]`` BF16
    contiguous (``V`` a multiple of 256, at most 1572864).
    """
    if backend != "cake":
        raise ValueError(
            "the candidate-support LM-head kernels currently support backend='cake'"
        )
    return _cake_support().chunked_lm_head_det_support_logprob(
        X,
        W,
        support_ids,
        support_offsets,
        labels,
        k_cap=k_cap,
        temperature=temperature,
        chunk_size=chunk_size,
        check_inputs=check_inputs,
        backend="cake",
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_det_support_kl(
    X: torch.Tensor,
    W: torch.Tensor,
    support_ids: torch.Tensor,
    support_offsets: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    k_cap: int,
    temperature: Union[None, float, torch.Tensor] = 1.0,
    return_logprobs: bool = False,
    chunk_size: Optional[int] = None,
    check_inputs: bool = False,
    backend: str = "cake",
):
    r"""Unreduced forward KL from a detached teacher over a ragged candidate support, SM100 / SM103 / SM107 only.

    ``KL_t = sum_j Q_j (lq_j - logp_j)`` over the candidates of row ``t`` with ``Q_j = exp(lq_j) > 0``,
    for the FP32 teacher log-probabilities ``teacher_logprobs [M]`` aligned with ``support_ids``
    (``requires_grad=False``; ``-inf`` entries drop out) and the student ``logp`` of
    :func:`chunked_lm_head_det_support_logprob` on the same support; ``0`` on a row of one candidate or
    none.  Returns the FP32 ``kl [T]``, or ``(kl, logp)`` with the differentiable packed student
    ``logp [M]`` when ``return_logprobs=True``.  Weighting, reduction and any ``tau ** 2`` factor
    belong to the caller.  Other arguments as in :func:`chunked_lm_head_det_support_logprob`.
    """
    if backend != "cake":
        raise ValueError(
            "the candidate-support LM-head kernels currently support backend='cake'"
        )
    return _cake_support().chunked_lm_head_det_support_kl(
        X,
        W,
        support_ids,
        support_offsets,
        teacher_logprobs,
        k_cap=k_cap,
        temperature=temperature,
        return_logprobs=return_logprobs,
        chunk_size=chunk_size,
        check_inputs=check_inputs,
        backend="cake",
    )


# Read once at import: a malformed value fails here, naming the variable, not
# on the first call.  Applied to the backend's cache by ``_cake_backend``.
_BINDING_CACHE_SETTINGS: Dict[str, Any] = _binding_cache_settings()


# ---------------------------------------------------------------------------
# The det-free entries: ``deterministic`` selects the package.
#
# ``deterministic=True`` (the default) calls the deterministic entry's body
# directly with the caller's own positional and keyword arguments (the body is
# the function below the deterministic entry's experimental wrapper, whose
# once-only warning this entry's own wrapper gives): no keyword is re-bound or
# re-packed, no allocation, no synchronization -- one identity test of host
# work.  Each entry's signature (``__signature__``) is the deterministic
# entry's, plus ``deterministic`` where that one has none.  ``deterministic=
# False`` runs the non-deterministic package,
# flashinfer.experimental.cake.cake_lm_head_loss, under the same argument
# checks and the same FLASHINFER_CAKE_LM_HEAD_LOSS_* fallbacks; while torch's
# deterministic algorithms are enabled (``torch.use_deterministic_algorithms(
# True)``, also with ``warn_only``) it runs the deterministic entry instead and
# warns once per entry point (an explicit dX ``k_slices`` pin above the
# deterministic entry's four slices is then dropped, with one more warning).
# Any other value is a ``TypeError``.
# ---------------------------------------------------------------------------

_det_loss = chunked_lm_head_det_loss.__wrapped__
_det_logprob = chunked_lm_head_det_logprob.__wrapped__
_det_preload = chunked_lm_head_det_preload.__wrapped__
_det_support_logprob = chunked_lm_head_det_support_logprob.__wrapped__
_det_support_kl = chunked_lm_head_det_support_kl.__wrapped__

# The entry points that already warned that torch's deterministic mode ran the
# deterministic implementation for ``deterministic=False``.
_DET_FALLBACK_WARNED: set = set()
# The entry points that already warned that the deterministic fallback dropped
# a dX ``k_slices`` pin it does not serve.
_DET_FALLBACK_PIN_WARNED: set = set()
# The dX slice counts the deterministic entries serve (the non-deterministic
# package's reduction serves more).
_DET_DX_K_SLICES_MAX = 4

# The flashinfer package directory: frames below it are the entries, their
# decorators and the packages.
_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


def _caller_stacklevel() -> int:
    """The ``stacklevel`` of the nearest frame outside the flashinfer package,
    for a warning issued by the function that calls this one (the decorators
    between the caller and the entry vary with the API logging level, so a
    fixed ``stacklevel`` could name a frame of flashinfer)."""
    frame = inspect.currentframe()
    level = 0
    while frame is not None:
        parent = frame.f_back
        level += 1
        if parent is not None and not os.path.abspath(
            parent.f_code.co_filename
        ).startswith(_PACKAGE_ROOT + os.sep):
            return level
        frame = parent
    return 3


def _runs_det(deterministic: object, entry: str) -> bool:
    """Whether a call whose ``deterministic`` is not ``True`` runs the
    deterministic entry: ``False`` -> ``torch.are_deterministic_algorithms_enabled()``
    (warning once per entry point when it is); any non-bool is a ``TypeError``."""
    if deterministic is not False:
        raise TypeError(
            f"{entry}: deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if not torch.are_deterministic_algorithms_enabled():
        return False
    if entry not in _DET_FALLBACK_WARNED:
        _DET_FALLBACK_WARNED.add(entry)
        warnings.warn(
            f"{entry}(deterministic=False) runs the deterministic implementation "
            f"(chunked_lm_head_det_*) because torch.use_deterministic_algorithms(True) "
            "is in effect",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
    return True


def _raise_named(exc: TypeError, inner: Any, entry: str) -> NoReturn:
    """Re-raise ``exc`` -- a ``TypeError`` raised by calling ``inner`` -- naming
    the public ``entry`` where Python's argument-binding message names
    ``inner`` (an unknown keyword, a missing or surplus argument: raised before
    ``inner``'s body ran, so the traceback ends in the caller's frame); any
    other ``TypeError`` propagates unchanged."""
    message, prefix = str(exc), f"{getattr(inner, '__name__', '')}() "
    tb = exc.__traceback__
    if tb is None or tb.tb_next is not None or not message.startswith(prefix):
        raise exc
    raise TypeError(f"{entry}() {message[len(prefix) :]}") from None


def _det_fallback_kwargs(entry: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The keyword arguments of a ``deterministic=False`` call that runs the
    deterministic entry (:func:`_runs_det`): an explicit dX ``k_slices`` pin
    above :data:`_DET_DX_K_SLICES_MAX` (the non-deterministic package serves
    more; given as ``gemm_tuning`` or through ``FLASHINFER_CAKE_LM_HEAD_LOSS_
    GEMM_TUNING``) is dropped -- that GEMM then takes the deterministic
    entry's own slice rule -- with one warning per entry point; any other
    knob, and any other value of the pin, is passed on unchanged."""
    tuning = _env_gemm_tuning(kwargs.get("gemm_tuning"))
    dx = (tuning or {}).get("dx")
    k = dx.get("k_slices") if isinstance(dx, dict) else None
    if (
        not isinstance(k, int)
        or isinstance(k, bool)
        or k <= _DET_DX_K_SLICES_MAX
        or tuning is None
    ):
        return kwargs
    rest = {op: knobs for op, knobs in tuning.items() if op != "dx"}
    kept = {name: value for name, value in dx.items() if name != "k_slices"}
    if kept:
        rest["dx"] = kept
    if entry not in _DET_FALLBACK_PIN_WARNED:
        _DET_FALLBACK_PIN_WARNED.add(entry)
        warnings.warn(
            f"{entry}(deterministic=False) under torch.use_deterministic_algorithms(True): "
            f"the deterministic implementation serves 1 .. {_DET_DX_K_SLICES_MAX} dX K slices, "
            f"so gemm_tuning dx k_slices={k} is dropped (its own slice rule applies)",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
    return dict(kwargs, gemm_tuning=rest)


@functools.cache
def _nd_backend() -> ModuleType:
    # The non-deterministic package, imported on its first use (a call with
    # ``deterministic=False``); its binding cache takes the same settings.
    from .experimental.cake.cake_lm_head_loss import cake_backend

    _configure_binding_cache(cake_backend, _BINDING_CACHE_SETTINGS)
    return cake_backend


@functools.cache
def _nd_backend_opt_in() -> ModuleType:
    _nd_backend()
    from .experimental.cake.cake_lm_head_loss import cake_backend_opt_in

    _configure_binding_cache(cake_backend_opt_in, _BINDING_CACHE_SETTINGS)
    return cake_backend_opt_in


@functools.cache
def _nd_support() -> ModuleType:
    _nd_backend()
    from .experimental.cake.cake_lm_head_loss import cake_support

    return cake_support


def _no_temperature(temperature: object) -> bool:
    """Whether ``temperature`` is no temperature for the deterministic entry: ``None`` or a Python number (not a
    ``bool``) whose FP32 value is 1.0."""
    if temperature is None:
        return True
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        return False
    try:
        return struct.unpack("<f", struct.pack("<f", temperature))[0] == 1.0
    except OverflowError:
        return False


def _with_mode(det_fn, *, before: str) -> inspect.Signature:
    """``det_fn``'s signature with the keyword-only ``deterministic: bool = True``
    inserted before the parameter ``before`` (the entries whose deterministic
    form has no mode keyword)."""
    params = list(inspect.signature(det_fn).parameters.values())
    at = [p.name for p in params].index(before)
    mode = inspect.Parameter(
        "deterministic", inspect.Parameter.KEYWORD_ONLY, default=True, annotation=bool
    )
    return inspect.signature(det_fn).replace(
        parameters=params[:at] + [mode] + params[at:]
    )


def _nd_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    deterministic: object,
    objective: str = "ce",
    loss_div: Optional[float] = None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
    return_logp: bool = False,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[Dict[str, Dict[str, Any]]] = None,
    memory_mode: Optional[str] = None,
    num_valid_rows: Optional[int] = None,
    check_num_valid_rows: bool = False,
):
    """:func:`chunked_lm_head_loss` for ``deterministic`` other than ``True``."""
    if _runs_det(deterministic, "chunked_lm_head_loss"):
        kwargs = dict(
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
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
        )
        return _det_loss(
            X, W, labels, **_det_fallback_kwargs("chunked_lm_head_loss", kwargs)
        )
    if backend != "cake":
        raise ValueError(
            "the chunked LM-head + loss kernels currently support backend='cake'"
        )
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_CHECK_WITHOUT_COUNT)
    knobs = _resolve_knobs(
        compact_rows=compact_rows,
        fuse_dw_cast=fuse_dw_cast,
        dx_finalize=dx_finalize,
        dw_stream=dw_stream,
        hidden_count=hidden_count,
        gemm_tuning=gemm_tuning,
    )
    if memory_mode is not None or num_valid_rows is not None:
        return _nd_backend_opt_in().chunked_lm_head_loss(
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
            deterministic=False,
            backend="cake",
            **knobs,
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
        )
    return _nd_backend().chunked_lm_head_loss(
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
        deterministic=False,
        backend="cake",
        **knobs,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_loss(*args: Any, deterministic: bool = True, **kwargs: Any):
    r"""Differentiable chunked LM-head projection + loss with a selectable
    numerical mode, SM100 / SM103 only.

    The computation, the arguments and the outputs are those of
    :func:`chunked_lm_head_det_loss` (each switch keyword left at ``None``
    reads the same ``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` variable in both modes).

    Parameters
    ----------
    deterministic : bool
        ``True`` (default): :func:`chunked_lm_head_det_loss` with the same
        arguments, passed on as given -- the same outputs bit for bit, with no
        added allocation or synchronization.  ``False``: the non-deterministic
        package (``flashinfer.experimental.cake.cake_lm_head_loss``): the same
        precision boundaries (BF16 logits and ``dlogits``, FP32 loss
        arithmetic, FP32 ``dX`` / ``dW`` accumulation, one final cast each),
        with each K slice of a sliced dX GEMM added straight into the FP32
        ``dX`` accumulator by an unordered FP32 reduction (no slabs), so
        ``dX`` is not bitwise reproducible run to run (``loss``, ``logp`` and
        ``dW`` keep their fixed order); ``dx_finalize`` then selects only the
        compacted ``dX`` output's one-pass scatter (there are no slabs to add).
        The gradient placement is that package's own process setting
        (``cake_backend.set_gradient_placement`` of
        ``flashinfer.experimental.cake.cake_lm_head_loss``, default
        ``"auto"``, the deterministic backend's rule).
        While ``torch.use_deterministic_algorithms(True)`` is in effect,
        ``deterministic=False`` runs the deterministic entry and warns once
        (a dX ``k_slices`` pin above its four slices is dropped, with one
        warning).  A value that is not a ``bool`` is a ``TypeError``.

    Every other parameter, the return value and the notes: see
    :func:`chunked_lm_head_det_loss`.
    """
    if deterministic is True:
        try:
            return _det_loss(*args, **kwargs)
        except TypeError as exc:
            _raise_named(exc, _det_loss, "chunked_lm_head_loss")
    try:
        return _nd_loss(*args, deterministic=deterministic, **kwargs)
    except TypeError as exc:
        _raise_named(exc, _nd_loss, "chunked_lm_head_loss")


chunked_lm_head_loss.__wrapped__.__signature__ = inspect.signature(_det_loss)


def _nd_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    deterministic: object,
    chunk_size: int = 4096,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[Dict[str, Dict[str, Any]]] = None,
    memory_mode: Optional[str] = None,
    num_valid_rows: Optional[int] = None,
    check_num_valid_rows: bool = False,
    temperature: Union[None, float, torch.Tensor] = None,
    check_temperature: bool = False,
) -> torch.Tensor:
    """:func:`chunked_lm_head_logprob` for ``deterministic`` other than ``True``."""
    if _runs_det(deterministic, "chunked_lm_head_logprob"):
        kwargs = dict(
            chunk_size=chunk_size,
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
            temperature=temperature,
            check_temperature=check_temperature,
        )
        return _det_logprob(
            X, W, labels, **_det_fallback_kwargs("chunked_lm_head_logprob", kwargs)
        )
    if not _no_temperature(temperature):
        # the non-deterministic package has no temperature forms
        raise ValueError(
            "chunked_lm_head_logprob(deterministic=False) takes no temperature (the "
            "non-deterministic package has no temperature path): pass temperature=None "
            "or 1.0, or deterministic=True"
        )
    if backend != "cake":
        raise ValueError(
            "the chunked LM-head + loss kernels currently support backend='cake'"
        )
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_CHECK_WITHOUT_COUNT)
    knobs = _resolve_knobs(
        compact_rows=compact_rows,
        fuse_dw_cast=fuse_dw_cast,
        dx_finalize=dx_finalize,
        dw_stream=dw_stream,
        hidden_count=hidden_count,
        gemm_tuning=gemm_tuning,
    )
    if memory_mode is not None or num_valid_rows is not None:
        return _nd_backend_opt_in().chunked_lm_head_logprob(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            deterministic=False,
            backend="cake",
            **knobs,
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
        )
    return _nd_backend().chunked_lm_head_logprob(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        deterministic=False,
        backend="cake",
        **knobs,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_logprob(
    *args: Any, deterministic: bool = True, **kwargs: Any
) -> torch.Tensor:
    r"""Differentiable chunked selected-token log-probability with a selectable
    numerical mode, SM100 / SM103 only.

    The computation and the arguments are those of
    :func:`chunked_lm_head_det_logprob`.  ``deterministic=True`` (default):
    that entry with the same arguments, passed on as given (the same bits, no
    added allocation or synchronization); ``deterministic=False``: the
    non-deterministic package (``flashinfer.experimental.cake.cake_lm_head_loss``),
    whose backward adds each K slice of a sliced dX GEMM into the FP32 ``dX``
    accumulator by an unordered FP32 reduction (``logp`` keeps its fixed
    order); under ``torch.use_deterministic_algorithms(True)`` it runs the
    deterministic entry and warns once (a dX ``k_slices`` pin above its four
    slices is dropped, with one warning).  A value that is not a ``bool`` is a
    ``TypeError``.  ``temperature`` is served by the deterministic entry only:
    with ``deterministic=False`` the non-deterministic package (no temperature
    path) takes ``None`` or a scalar whose FP32 value is 1.0 (the call without a
    temperature), and any other value is a ``ValueError``.
    """
    if deterministic is True:
        try:
            return _det_logprob(*args, **kwargs)
        except TypeError as exc:
            _raise_named(exc, _det_logprob, "chunked_lm_head_logprob")
    try:
        return _nd_logprob(*args, deterministic=deterministic, **kwargs)
    except TypeError as exc:
        _raise_named(exc, _nd_logprob, "chunked_lm_head_logprob")


chunked_lm_head_logprob.__wrapped__.__signature__ = inspect.signature(_det_logprob)


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_preload(
    device: Optional[Union[torch.device, int, str]] = None,
    *,
    deterministic: bool = True,
    temperature: bool = False,
):
    r"""Do part of the one-time work of a process's first chunked LM-head call
    ahead of it, then wait for the device once (SM100 / SM103 only).

    ``deterministic=True`` (default): :func:`chunked_lm_head_det_preload`
    (the deterministic package's libraries and per-device state);
    ``deterministic=False``: the same for the non-deterministic package
    (``flashinfer.experimental.cake.cake_lm_head_loss``), or the deterministic
    package's while ``torch.use_deterministic_algorithms(True)`` is in effect
    (the package a ``deterministic=False`` call then runs; one warning).  The
    per-thread state (side streams, events, count cells) is the calling
    thread's: another thread forms its own at its first call.  A value that is
    not a ``bool`` is a ``TypeError``.  ``temperature=True`` also loads the
    deterministic package's temperature libraries
    (:func:`chunked_lm_head_det_preload`); the non-deterministic package has no
    temperature path, so ``deterministic=False`` with ``temperature=True`` is a
    ``ValueError``.
    """
    if deterministic is True or _runs_det(deterministic, "chunked_lm_head_preload"):
        if temperature is False:
            return _det_preload(device)
        return _det_preload(device, temperature=temperature)
    if temperature is not False:
        raise ValueError(
            "chunked_lm_head_preload(deterministic=False) takes no temperature (the "
            "non-deterministic package has no temperature path): pass temperature=False, "
            "or deterministic=True"
        )
    _nd_backend_opt_in().preload(device)


def _nd_support_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    support_ids: torch.Tensor,
    support_offsets: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    *,
    k_cap: int,
    temperature: Union[None, float, torch.Tensor] = 1.0,
    chunk_size: Optional[int] = None,
    check_inputs: bool = False,
    deterministic: object = False,
    backend: str = "cake",
) -> torch.Tensor:
    """:func:`chunked_lm_head_support_logprob` for ``deterministic`` other than ``True``."""
    if _runs_det(deterministic, "chunked_lm_head_support_logprob"):
        return _det_support_logprob(
            X,
            W,
            support_ids,
            support_offsets,
            labels,
            k_cap=k_cap,
            temperature=temperature,
            chunk_size=chunk_size,
            check_inputs=check_inputs,
            backend=backend,
        )
    if backend != "cake":
        raise ValueError(
            "the candidate-support LM-head kernels currently support backend='cake'"
        )
    return _nd_support().chunked_lm_head_support_logprob(
        X,
        W,
        support_ids,
        support_offsets,
        labels,
        k_cap=k_cap,
        temperature=temperature,
        chunk_size=chunk_size,
        check_inputs=check_inputs,
        deterministic=False,
        backend="cake",
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_support_logprob(
    *args: Any, deterministic: bool = True, **kwargs: Any
) -> torch.Tensor:
    r"""Differentiable log-probabilities over a ragged candidate support per
    token with a selectable numerical mode, SM100 / SM103 only.

    The computation and the other arguments are those of
    :func:`chunked_lm_head_det_support_logprob`.  ``deterministic=True``
    (default): that entry with the same arguments, passed on as given (the
    same bits, no added allocation or synchronization); ``deterministic=False``:
    the non-deterministic package's entry
    (``flashinfer.experimental.cake.cake_lm_head_loss.cake_support``), which
    makes no run-to-run bitwise promise for the gradients; under
    ``torch.use_deterministic_algorithms(True)`` it runs the deterministic
    entry and warns once.  A value that is not a ``bool`` is a ``TypeError``.
    """
    if deterministic is True:
        try:
            return _det_support_logprob(*args, **kwargs)
        except TypeError as exc:
            _raise_named(exc, _det_support_logprob, "chunked_lm_head_support_logprob")
    try:
        return _nd_support_logprob(*args, deterministic=deterministic, **kwargs)
    except TypeError as exc:
        _raise_named(exc, _nd_support_logprob, "chunked_lm_head_support_logprob")


chunked_lm_head_support_logprob.__wrapped__.__signature__ = _with_mode(
    _det_support_logprob, before="backend"
)


def _nd_support_kl(
    X: torch.Tensor,
    W: torch.Tensor,
    support_ids: torch.Tensor,
    support_offsets: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    k_cap: int,
    temperature: Union[None, float, torch.Tensor] = 1.0,
    return_logprobs: bool = False,
    chunk_size: Optional[int] = None,
    check_inputs: bool = False,
    deterministic: object = False,
    backend: str = "cake",
):
    """:func:`chunked_lm_head_support_kl` for ``deterministic`` other than ``True``."""
    if _runs_det(deterministic, "chunked_lm_head_support_kl"):
        return _det_support_kl(
            X,
            W,
            support_ids,
            support_offsets,
            teacher_logprobs,
            k_cap=k_cap,
            temperature=temperature,
            return_logprobs=return_logprobs,
            chunk_size=chunk_size,
            check_inputs=check_inputs,
            backend=backend,
        )
    if backend != "cake":
        raise ValueError(
            "the candidate-support LM-head kernels currently support backend='cake'"
        )
    return _nd_support().chunked_lm_head_support_kl(
        X,
        W,
        support_ids,
        support_offsets,
        teacher_logprobs,
        k_cap=k_cap,
        temperature=temperature,
        return_logprobs=return_logprobs,
        chunk_size=chunk_size,
        check_inputs=check_inputs,
        deterministic=False,
        backend="cake",
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_support_kl(*args: Any, deterministic: bool = True, **kwargs: Any):
    r"""Unreduced forward KL from a detached teacher over a ragged candidate
    support with a selectable numerical mode, SM100 / SM103 only.

    The computation and the other arguments are those of
    :func:`chunked_lm_head_det_support_kl`; ``deterministic`` as in
    :func:`chunked_lm_head_support_logprob`.
    """
    if deterministic is True:
        try:
            return _det_support_kl(*args, **kwargs)
        except TypeError as exc:
            _raise_named(exc, _det_support_kl, "chunked_lm_head_support_kl")
    try:
        return _nd_support_kl(*args, deterministic=deterministic, **kwargs)
    except TypeError as exc:
        _raise_named(exc, _nd_support_kl, "chunked_lm_head_support_kl")


chunked_lm_head_support_kl.__wrapped__.__signature__ = _with_mode(
    _det_support_kl, before="backend"
)
