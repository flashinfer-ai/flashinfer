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

Cake deterministic LM-head + loss training entries (SM100 / SM103 / SM107):
the chunked LM-head projection with a cross-entropy or policy objective, its
per-row log-probabilities, and the log-probabilities / KL over a ragged
candidate support.  Fixed sequential chunk order, no atomics: outputs and
gradients are bitwise reproducible run to run.

Entries: :func:`cake_lm_head_det_train_loss`,
:func:`cake_lm_head_det_train_logprob`, :func:`cake_lm_head_det_train_preload`,
:func:`cake_lm_head_det_train_support_logprob` and
:func:`cake_lm_head_det_train_support_kl`.  The mode is part of the name: no
entry takes a ``deterministic`` keyword (passing one is a ``TypeError``), and
``torch.use_deterministic_algorithms(True)`` changes nothing (these entries
never raise its alert).

A stable API (``@flashinfer_api``; no ``ExperimentalWarning``) of thin
wrappers: validation of the backend contract, the chunk plans, the workspace,
launch binding, autograd wrappers and JIT registration live in the
experimental backend package ``flashinfer.experimental.cake.cake_lm_head_det_train``,
imported on first use.  The ``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` variables are
this wrapper's A/B knobs (``flashinfer/_cake_lm_head_train_knobs.py``): a
switch keyword left at ``None`` falls back to its variable, and the resolved
value is passed down.  The two binding-cache variables are read once, when
this module is imported, and applied to the backend's cache on first use (the
first call of any entry).
"""

import functools
from types import ModuleType
from typing import Any, Dict, Optional, Union

import torch

from . import _cake_lm_head_train_knobs as _knobs
from .api_logging import flashinfer_api

__all__ = [
    "cake_lm_head_det_train_logprob",
    "cake_lm_head_det_train_loss",
    "cake_lm_head_det_train_preload",
    "cake_lm_head_det_train_support_kl",
    "cake_lm_head_det_train_support_logprob",
]


@functools.cache
def _cake_backend() -> ModuleType:
    # The backend is imported on first use (experimental containment: core
    # reaches kernels and their dependencies through a deferred import) and
    # the binding-cache settings read at import are applied to it once.
    from .experimental.cake.cake_lm_head_det_train import cake_backend

    _knobs.configure_binding_cache(cake_backend, _BINDING_CACHE_SETTINGS)
    return cake_backend


@functools.cache
def _cake_backend_opt_in() -> ModuleType:
    # The opt-in forms (``memory_mode`` / ``num_valid_rows``) and the default
    # call's automatic memory form, imported on their first use; a call that
    # takes none of them (:func:`_opt_in_call`) never imports them.  Their own
    # binding cache takes the same settings.
    _cake_backend()
    from .experimental.cake.cake_lm_head_det_train import cake_backend_opt_in

    _knobs.configure_binding_cache(cake_backend_opt_in, _BINDING_CACHE_SETTINGS)
    return cake_backend_opt_in


def _opt_in_call(W, memory_mode, num_valid_rows) -> bool:
    # A call with ``memory_mode`` or ``num_valid_rows`` takes the opt-in
    # forms; so does a default call whose ``W`` requires a gradient: with grad
    # mode on its automatic form (``memory_mode=None``) can pick a budget
    # form, and with grad mode off (e.g. a reentrant checkpoint's first pass)
    # its hidden valid-row count is checked against the step's bound.  Every
    # other call runs the default entry.
    return (
        memory_mode is not None
        or num_valid_rows is not None
        or (isinstance(W, torch.Tensor) and W.requires_grad)
    )


def _kept_logits_reach(X, labels, chunk_size) -> bool:
    # A default log-probability call without a temperature whose ``X`` requires
    # a gradient under grad mode and whose rows fit two chunks also takes the
    # opt-in forms when its ``W`` is frozen: their one-chunk kept logits
    # (``cake_backend_opt_in._one_chunk_keep_rule``, the rule) can apply to it.
    # A pre-filter only: a call the rule does not keep runs the default entry's
    # form there.
    return (
        isinstance(X, torch.Tensor)
        and X.is_cuda
        and X.requires_grad
        and torch.is_grad_enabled()
        and isinstance(labels, torch.Tensor)
        and isinstance(chunk_size, int)
        and 0 < labels.numel() <= 2 * chunk_size
    )


@flashinfer_api
def cake_lm_head_det_train_loss(
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
        ``None`` (the default): the automatic form.  A call that forms a BF16
        ``dW`` on CUDA (grad mode on, ``W`` requiring a gradient) runs the
        fastest form while its predicted step peak, with that form's own
        buffer lifetimes, stays within ``B`` less the 2 % margin below, else
        the ``"budget"`` form the planner picks when one fits, else the
        fastest form (``cake_backend_opt_in.auto_memory_plan``).  A call whose
        valid rows fit one chunk keeps the fastest form: with the fused
        weight-gradient cast (``fuse_dw_cast``, on by default) it holds no
        FP32 ``[V, H]`` accumulator (without it, it does), and its step peak
        can exceed ``B`` (see the package README).  It plans on
        the valid rows of the forward's count or of ``num_valid_rows``; an
        uncompacted call without ``num_valid_rows`` keeps the fastest form,
        as does every call without such a ``dW``.  ``"fast"``: always the
        fastest form, whose FP32 ``[V, H]`` weight-gradient accumulator lives
        next to the chunk buffers; it first goes through the opt-in dispatch
        (``cake_backend_opt_in``) and then runs the backend's default entry.
        ``"budget"``: bound the step peak (forward through backward, outputs
        included) by the bytes ``B`` of a head that materializes the BF16
        logits of the valid rows (``B = 2 * T_v * V + 2 * V * H + 2 * T * H``);
        the planner targets ``B`` less a 2 % margin: per call, the fastest
        exact weight-gradient form whose predicted peak fits -- the fused or
        the unfused cast, or the forward accumulating the leading vocabulary
        rows and the backward recomputing the others in vocabulary blocks
        (``cake_backend_opt_in.memory_plan``).  Bitwise the same ``loss`` /
        ``logp`` / ``dX`` / ``dW`` as ``"fast"``; the recompute forms run more
        GEMM work.  A default call whose automatic form is a budget form gives
        the same bits, except that a call with zero-weight policy rows decides
        its row split as ``"budget"`` does (``dX`` / ``dW`` within the
        accuracy budget; see the package README).  The bound assumes a
        backward that frees the graph (``retain_graph=True`` keeps the
        forward's state, also for a default call); its margin is measured on
        GB200, GB300 and B200 (see the package README).  ``"keep_logits"``
        belongs to ``cake_lm_head_det_train_logprob`` (a ``ValueError``
        here).  No environment variable selects the mode; the switches above
        (``compact_rows``, ``fuse_dw_cast``, ``dx_finalize``, ``hidden_count``,
        ``gemm_tuning``, and their ``FLASHINFER_CAKE_LM_HEAD_LOSS_*``
        variables) are inputs of the automatic form's rule.  The
        non-deterministic module ``flashinfer.cake_lm_head_nondet_train``,
        whose entries refer to this description, has no automatic form: its
        default ``None`` is the fastest form.
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
    Fixed sequential chunk order, no atomics: ``loss``, ``logp``, ``dX`` and
    ``dW`` are bitwise reproducible run to run.  The det name is the mode, so
    there is no ``deterministic`` keyword (passing one is a ``TypeError``);
    torch's deterministic mode (``torch.use_deterministic_algorithms``)
    changes nothing.

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

    A call with neither ``memory_mode`` nor ``num_valid_rows`` whose ``W``
    requires no gradient runs the backend's default entry (``cake_backend``)
    after this function's argument checks (``check_num_valid_rows``, then
    ``memory_mode`` and ``num_valid_rows`` against ``None``, then
    ``W.requires_grad``).  Every other call goes through
    ``cake_backend_opt_in``: an explicit ``memory_mode="fast"`` without a
    count, and a default call whose step forms no BF16 ``dW`` on the cake
    backend on CUDA, then run the same default entry; a default call that
    can switch (grad mode on) runs this entry's autograd function of
    ``cake_backend_opt_in`` (its fast form bitwise the default entry's); a
    default call under grad mode off whose ``W`` requires a gradient (e.g.
    the first pass of a reentrant ``torch.utils.checkpoint``) runs the fast
    form's forward with its hidden valid-row count checked against the
    step's bound (the same bits); every other such call takes its opt-in
    forms.  With ``memory_mode="budget"``, or a default call whose
    automatic form is a budget form, a backward that frees the graph releases
    the saved accumulators as soon as it has used them (a second backward
    then raises).

    ``cake_lm_head_det_train_preload`` loads the libraries and creates the
    per-device state ahead of a process's first call.

    Host side: inputs are validated and bound on the first call for an input
    binding (``data_ptr``, shape, stride and dtype of every input plus the
    options); later calls with the same binding launch from the remembered
    argument plans
    (``flashinfer.experimental.cake.cake_lm_head_det_train.cake_backend.BINDING_CACHE``;
    the calls through ``cake_backend_opt_in`` -- the default loss calls that
    form a weight gradient among them, and a log-probability backward in a
    budget form -- keep theirs in ``cake_backend_opt_in.BINDING_CACHE``).
    ``FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE=0`` disables both caches and
    ``FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY`` (a positive
    integer; default 64) sets their capacity: both are read when this module
    is imported and applied through ``BINDING_CACHE.configure`` on the first
    call that reaches each backend module; a later
    ``cake_backend.BINDING_CACHE.configure(...)`` sets that cache only.  Peak
    temporary memory is reported separately from the weights, the outputs and
    the FP32 accumulators by ``cake_backend.memory_report``.

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
        raise ValueError(_knobs.CHECK_WITHOUT_COUNT)
    if _opt_in_call(W, memory_mode, num_valid_rows):
        return _cake_backend_opt_in().cake_lm_head_det_train_loss(
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
            backend="cake",
            **_knobs.resolve_knobs(
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
    return _cake_backend().cake_lm_head_det_train_loss(
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
        backend="cake",
        **_knobs.resolve_knobs(
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
        ),
    )


def _logprob_temperature(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
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
    """:func:`cake_lm_head_det_train_logprob` with a ``temperature`` (not ``None``): the same routing, through the
    backends' ``cake_lm_head_det_train_logprob_temperature`` entries."""
    if backend != "cake":
        raise ValueError(
            "the chunked LM-head + loss kernels currently support backend='cake'"
        )
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_knobs.CHECK_WITHOUT_COUNT)
    if _opt_in_call(W, memory_mode, num_valid_rows):
        return _cake_backend_opt_in().cake_lm_head_det_train_logprob_temperature(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            backend="cake",
            **_knobs.resolve_knobs(
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
    return _cake_backend().cake_lm_head_det_train_logprob_temperature(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        backend="cake",
        **_knobs.resolve_knobs(
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


@flashinfer_api
def cake_lm_head_det_train_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
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
    ``dW`` from its epilogue).  Arguments as in :func:`cake_lm_head_det_train_loss`,
    including the switches ``compact_rows``, ``fuse_dw_cast``, ``dx_finalize``,
    ``dw_stream``, ``hidden_count`` and ``gemm_tuning`` with the same defaults
    (``None`` reads the same ``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` variable; every
    switch is resolved once in the forward and the backward recomputes under
    the same values); ``dW`` is returned in BF16.  ``memory_mode``
    (``"budget"``: the backward's weight-gradient form within the same bound;
    the default ``None``: the automatic form, decided in the backward, which
    keeps the default entry's forward; a default call without a temperature,
    of at most two chunks of rows, whose valid rows fit one chunk keeps their
    BF16 logits, at most one chunk, from the forward to its backward, which
    forms the logit gradient from them
    instead of recomputing the logits, where the fast backward would run no
    zero-gradient row skip -- not under ``torch.utils.checkpoint``, bitwise the
    same outputs; ``"fast"`` turns this off), ``num_valid_rows`` and
    ``check_num_valid_rows`` (the forward's count) as in
    :func:`cake_lm_head_det_train_loss`.  A default call whose backward runs
    a budget form runs it without the zero-gradient row skip: its ``dX`` /
    ``dW`` bits can then differ from the fastest form's where valid rows have
    a zero upstream gradient (within the accuracy budget, the same bits on
    every run).  ``memory_mode="keep_logits"`` (this
    entry only; opt-in; with ``num_valid_rows`` its forward runs on the index
    formed from the caller's count, without a readback): where a backward can
    follow (grad mode on, ``X`` or ``W`` trainable) the forward keeps the BF16
    logits of the valid rows, ``2 * T_v * V`` bytes from the forward to the
    backward (the chunk bound of the memory rule does not hold for them), and
    the first backward forms the logit gradient from them instead of
    recomputing the logits (three GEMMs per chunk instead of four; a second
    backward through a retained graph recomputes); where none can follow
    (``torch.no_grad``, ``torch.inference_mode``, frozen inputs) the call is
    the default one.  Bitwise the same ``logp``, ``dX`` and ``dW``.  As for
    :func:`cake_lm_head_det_train_loss`, the det name is the mode: there is no
    ``deterministic`` keyword (``TypeError``) and torch's deterministic mode
    changes nothing.

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

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, H]`` hidden states, with the layout rules of :func:`cake_lm_head_det_train_loss`.
    W : torch.Tensor
        Contiguous BF16 ``[V, H]`` output weight.
    labels : torch.Tensor
        int64 ``[T]`` selected token ids; ``-100`` marks an ignored row.
    chunk_size : int, default=4096
        Maximum token rows per logits chunk.
    backend : str, default="cake"
        Only ``"cake"`` is supported.
    compact_rows, fuse_dw_cast, dx_finalize, dw_stream, hidden_count, gemm_tuning
        Execution switches with the contracts and defaults of :func:`cake_lm_head_det_train_loss`.
    memory_mode : str, optional
        Backward memory policy described above; this entry also accepts ``"keep_logits"``.
    num_valid_rows : int, optional
        Caller-supplied host count of labels other than ``-100``.
    check_num_valid_rows : bool, default=False
        Read the valid-row count back and raise ``ValueError`` on a mismatch.
    temperature : float or torch.Tensor, optional
        Positive finite scalar or FP32 ``[T]`` temperature; ``None`` is equivalent to 1.0.
    check_temperature : bool, default=False
        Read tensor-temperature validation back and raise ``ValueError`` instead of a device assertion.
    """
    if temperature is not None:
        return _logprob_temperature(
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
        raise ValueError(_knobs.CHECK_WITHOUT_COUNT)
    if _opt_in_call(W, memory_mode, num_valid_rows) or _kept_logits_reach(
        X, labels, chunk_size
    ):
        return _cake_backend_opt_in().cake_lm_head_det_train_logprob(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            backend="cake",
            **_knobs.resolve_knobs(
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
    return _cake_backend().cake_lm_head_det_train_logprob(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        backend="cake",
        **_knobs.resolve_knobs(
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
        ),
    )


@flashinfer_api
def cake_lm_head_det_train_preload(
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
    :func:`cake_lm_head_det_train_logprob` temperature path and launches its PyTorch
    operators once; by default they are not loaded (callers that never pass
    a temperature build no temperature library).

    Parameters
    ----------
    device : torch.device, int, str, optional
        CUDA device whose libraries to load; ``None`` selects the current device.
    temperature : bool, default=False
        Also load the temperature-path libraries and warm up their PyTorch operators.
    """
    _cake_backend_opt_in().cake_lm_head_det_train_preload(
        device, temperature=temperature
    )


@functools.cache
def _cake_support() -> ModuleType:
    # The candidate-support entries live in their own module of the package; it
    # binds its kernels through the backend's cache, configured on first use.
    _cake_backend()
    from .experimental.cake.cake_lm_head_det_train import cake_support

    return cake_support


@flashinfer_api
def cake_lm_head_det_train_support_logprob(
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

    Parameters
    ----------
    X : torch.Tensor
        Row-major BF16 ``[T, H]`` hidden states, with ``H`` a multiple of 256.
    W : torch.Tensor
        Contiguous BF16 ``[V, H]`` output weight, with ``V`` a multiple of 256.
    support_ids : torch.Tensor
        Packed int32 or int64 candidate ids ``[M]``, unique within each row.
    support_offsets : torch.Tensor
        int32 or int64 ``[T + 1]`` row offsets, first 0 and last ``M``.
    labels : torch.Tensor, optional
        int64 ``[T]`` selected candidate ids (``-100`` ignores a row); ``None`` returns all candidate log-probabilities.
    k_cap : int
        Maximum candidates per row; rows exceeding it are rejected, not trimmed.
    temperature : float or torch.Tensor, optional
        Positive finite scalar or FP32 ``[T]`` per-row temperature; default 1.0.
    chunk_size : int, optional
        Token chunk size for the dense pass and large-support weight gradient; default 4096.
    check_inputs : bool, default=False
        Synchronize validation and raise ``ValueError`` on invalid input; otherwise use a device assertion.
    backend : str, default="cake"
        Only ``"cake"`` is supported.
    """
    if backend != "cake":
        raise ValueError(
            "the candidate-support LM-head kernels currently support backend='cake'"
        )
    return _cake_support().cake_lm_head_det_train_support_logprob(
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


@flashinfer_api
def cake_lm_head_det_train_support_kl(
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
    :func:`cake_lm_head_det_train_support_logprob` on the same support; ``0`` on a row of one candidate or
    none.  Returns the FP32 ``kl [T]``, or ``(kl, logp)`` with the differentiable packed student
    ``logp [M]`` when ``return_logprobs=True``.  Weighting, reduction and any ``tau ** 2`` factor
    belong to the caller.  Other arguments as in :func:`cake_lm_head_det_train_support_logprob`.

    Parameters
    ----------
    X : torch.Tensor
        Row-major BF16 ``[T, H]`` hidden states, with ``H`` a multiple of 256.
    W : torch.Tensor
        Contiguous BF16 ``[V, H]`` output weight, with ``V`` a multiple of 256.
    support_ids : torch.Tensor
        Packed int32 or int64 candidate ids ``[M]``, unique within each row.
    support_offsets : torch.Tensor
        int32 or int64 ``[T + 1]`` row offsets, first 0 and last ``M``.
    teacher_logprobs : torch.Tensor
        Detached FP32 ``[M]`` teacher log-probabilities aligned with ``support_ids``.
    return_logprobs : bool, default=False
        Also return the differentiable packed student log-probabilities.
    k_cap : int
        Maximum candidates per row; rows exceeding it are rejected, not trimmed.
    temperature : float or torch.Tensor, optional
        Positive finite scalar or FP32 ``[T]`` per-row temperature; default 1.0.
    chunk_size : int, optional
        Token chunk size for the dense pass and large-support weight gradient; default 4096.
    check_inputs : bool, default=False
        Synchronize validation and raise ``ValueError`` on invalid input; otherwise use a device assertion.
    backend : str, default="cake"
        Only ``"cake"`` is supported.
    """
    if backend != "cake":
        raise ValueError(
            "the candidate-support LM-head kernels currently support backend='cake'"
        )
    return _cake_support().cake_lm_head_det_train_support_kl(
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
_BINDING_CACHE_SETTINGS: Dict[str, Any] = _knobs.binding_cache_settings()
