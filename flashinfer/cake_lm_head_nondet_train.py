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

Cake non-deterministic LM-head + loss training entries (SM100 / SM103 /
SM107): the chunked LM-head projection with a cross-entropy or policy
objective, its per-row log-probabilities, and the log-probabilities / KL over
a ragged candidate support.  The same computation, arguments and precision
boundaries as the deterministic module ``flashinfer.cake_lm_head_det_train``.
A sliced dX GEMM stores each K slice into an FP32 slab, and a slab sum adds the
slices in slice order and stores the sum into the FP32 ``dX`` accumulator: one
fixed order, so ``dX`` repeats bit for bit run to run, and its bits may differ
from the deterministic module's by the rounding of that sum (``loss``,
``logp`` and ``dW`` keep their fixed order).  Even so, the module makes no
run-to-run bitwise promise for the gradients; the deterministic module makes
it.  Two row skips of this backend also differ from the deterministic
module: zero-weight policy rows and log-probability rows with a zero ``dlogp``
leave the gradient GEMMs, so ``loss`` / ``dW`` bits may differ from the
deterministic module's within the accuracy budgets, and a non-finite value in
a zero-weight row does not reach ``loss`` / ``dX`` / ``dW``.  By default a
one-row cross-entropy loss call at ``H = 6144`` (BF16 contiguous ``W``, one
chunk, compacted rows, no explicit ``gemm_tuning``, a registered program; the
conditions of ``cake_backend.set_one_pass_rows``) runs the backend's one-pass
programs: its ``loss`` / ``logp`` / ``dW`` are fixed order and bitwise
reproducible run to run, but are not the deterministic module's bits (within
the accuracy budgets).  Any other one-row call runs the chunked path.  The
backend README lists the output differences ("Zero-weight policy rows",
"Zero-gradient rows of the log-probability backward" and the
``set_one_pass_rows`` switch).

Entries: :func:`cake_lm_head_nondet_train_loss`,
:func:`cake_lm_head_nondet_train_logprob`,
:func:`cake_lm_head_nondet_train_preload`,
:func:`cake_lm_head_nondet_train_support_logprob` and
:func:`cake_lm_head_nondet_train_support_kl`.  The mode is part of the name:
no entry takes a ``deterministic`` keyword (passing one is a ``TypeError``).
Under ``torch.use_deterministic_algorithms(True)`` the backend raises
PyTorch's alert right before a K-sliced dX reduction (a ``RuntimeError``;
with ``warn_only=True`` a ``UserWarning``, and the call runs on); the alert
is part of this module's contract, as it makes no run-to-run bitwise promise
for the gradients.  A call without a K-sliced dX reduction does not alert.
Nothing falls back to the deterministic module: its entries are
``flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_*``.

A stable API (``@flashinfer_api``; no ``ExperimentalWarning``) of thin
wrappers: validation of the backend contract, the chunk plans, the workspace,
launch binding, autograd wrappers and JIT registration live in the
experimental backend package
``flashinfer.experimental.cake.cake_lm_head_nondet_train``, imported on first
use.  The ``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` variables are this wrapper's A/B
knobs (``flashinfer/_cake_lm_head_train_knobs.py``, shared with the
deterministic module): a switch keyword left at ``None`` falls back to its
variable, and the resolved value is passed down (except ``hidden_count``
with ``memory_mode="keep_logits"``; see
:func:`cake_lm_head_nondet_train_logprob`).  The two binding-cache
variables are read once, when this module is imported, and applied to the
backend's cache on first use (the first call of any entry).
"""

import functools
import os
import struct
from types import ModuleType
from typing import Any, Dict, Optional, Union

import torch

from . import _cake_lm_head_train_knobs as _knobs
from .api_logging import flashinfer_api

__all__ = [
    "cake_lm_head_nondet_train_logprob",
    "cake_lm_head_nondet_train_loss",
    "cake_lm_head_nondet_train_preload",
    "cake_lm_head_nondet_train_support_kl",
    "cake_lm_head_nondet_train_support_logprob",
]


@functools.cache
def _cake_backend() -> ModuleType:
    # The backend is imported on first use (experimental containment: core
    # reaches kernels and their dependencies through a deferred import) and
    # the binding-cache settings read at import are applied to it once.
    from .experimental.cake.cake_lm_head_nondet_train import cake_backend

    _knobs.configure_binding_cache(cake_backend, _BINDING_CACHE_SETTINGS)
    return cake_backend


@functools.cache
def _cake_backend_opt_in() -> ModuleType:
    # The opt-in forms (``memory_mode`` / ``num_valid_rows``) and the preload,
    # imported on their first use.  Their own binding cache takes the same
    # settings.
    _cake_backend()
    from .experimental.cake.cake_lm_head_nondet_train import cake_backend_opt_in

    _knobs.configure_binding_cache(cake_backend_opt_in, _BINDING_CACHE_SETTINGS)
    return cake_backend_opt_in


@functools.cache
def _cake_support() -> ModuleType:
    # The candidate-support entries live in their own module of the package; it
    # binds its kernels through the backend's cache, configured on first use.
    _cake_backend()
    from .experimental.cake.cake_lm_head_nondet_train import cake_support

    return cake_support


_KEEP_LOGITS = "keep_logits"


def _kept_hidden_count() -> Optional[bool]:
    # ``hidden_count`` of a ``memory_mode="keep_logits"`` call that leaves it
    # at ``None``: ``False`` when the variable is "0", else ``None`` (the
    # backend's kept-logits form, ``set_kept_logits_form``, decides).
    return False if os.environ.get(_knobs.HIDDEN_COUNT_ENV, "1") == "0" else None


@flashinfer_api
def cake_lm_head_nondet_train_loss(
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
    backward and a slab-summed dX reduction, SM100 / SM103 / SM107 only.

    See :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss`
    for the computation, the arguments, the switches and their
    ``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` fallbacks, ``memory_mode``,
    ``num_valid_rows`` and the outputs, except as the module docstring notes
    (the dX reduction, the zero-weight row split and the one-pass call).

    Notes
    -----
    The same precision boundaries (BF16 logits and ``dlogits``, FP32 loss
    arithmetic, FP32 ``dX`` / ``dW`` accumulation, one final cast each).  A
    sliced dX GEMM stores each K slice into an FP32 slab, and a slab sum adds
    the slices in slice order and stores the sum into the FP32 ``dX``
    accumulator, so ``dX`` repeats bit for bit run to run; ``loss``, ``logp``
    and ``dW`` keep their fixed order.  ``dx_finalize`` then selects only the
    compacted ``dX`` output's one-pass scatter (each chunk's slab sum has
    already added its slices).  The gradient placement is the backend's own
    process setting (``cake_backend.set_gradient_placement`` of
    ``flashinfer.experimental.cake.cake_lm_head_nondet_train``, default
    ``"auto"``, the deterministic backend's rule).

    There is no ``deterministic`` keyword (passing one is a ``TypeError``).
    While ``torch.use_deterministic_algorithms(True)`` is in effect the call
    still runs this backend: PyTorch's alert is raised before a K-sliced dX
    reduction (``RuntimeError``; with ``warn_only=True`` a ``UserWarning``,
    then the call runs on); there is no fallback to the deterministic entry,
    and every knob (a dX ``k_slices`` pin included) reaches the backend
    unchanged.

    A call with neither ``memory_mode`` nor ``num_valid_rows`` runs the
    backend's default entry (``cake_backend``) after this function's argument
    checks (``backend``, then ``check_num_valid_rows``); a call with either
    goes through ``cake_backend_opt_in``.

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
    knobs = _knobs.resolve_knobs(
        compact_rows=compact_rows,
        fuse_dw_cast=fuse_dw_cast,
        dx_finalize=dx_finalize,
        dw_stream=dw_stream,
        hidden_count=hidden_count,
        gemm_tuning=gemm_tuning,
    )
    if memory_mode is not None or num_valid_rows is not None:
        return _cake_backend_opt_in().cake_lm_head_nondet_train_loss(
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
            **knobs,
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
        )
    return _cake_backend().cake_lm_head_nondet_train_loss(
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
        **knobs,
    )


def _no_temperature(temperature: object) -> bool:
    """Whether ``temperature`` is no temperature (the call then runs without one): ``None`` or a Python number (not
    a ``bool``) whose FP32 value is 1.0."""
    if temperature is None:
        return True
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        return False
    try:
        return struct.unpack("<f", struct.pack("<f", temperature))[0] == 1.0
    except OverflowError:
        return False


@flashinfer_api
def cake_lm_head_nondet_train_logprob(
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
    r"""Differentiable chunked selected-token log-probability with a
    slab-summed dX reduction, SM100 / SM103 / SM107 only.

    See :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_logprob`
    for the computation and the arguments (``memory_mode="keep_logits"``,
    ``temperature`` and ``check_temperature`` included), except as the module
    docstring notes (the dX reduction and the zero-gradient row skip).
    The backward sums the K slices of a sliced dX GEMM in slice order and
    stores the sum into the FP32 ``dX`` accumulator (``logp`` keeps its fixed
    order); torch's deterministic mode raises PyTorch's alert before that
    reduction, as for :func:`cake_lm_head_nondet_train_loss`.  With a
    temperature the backend runs the same temperature forms (``logp`` the
    deterministic entry's bits, ``dW`` too when no valid row has a zero
    ``dlogp``, except a folded short tail and a call of the
    vocabulary-major backward, by default 2 to 4 chunks: another ``dW``
    grouping within the accuracy budgets; ``dX`` the same products, a sliced
    chunk's summed in slice order).  The zero-gradient row skip runs the dX / dW GEMMs of the rows
    with a nonzero ``dlogp`` only, with or without a temperature.  ``None``
    and a Python number whose FP32 value is 1.0 run the call without a
    temperature; a ``bool`` is a ``ValueError``.  With
    ``memory_mode="keep_logits"`` a ``hidden_count`` left at ``None`` is
    passed down as ``None`` (the backend's kept-logits form decides, see
    its ``set_kept_logits_form``) unless
    ``FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT`` is ``"0"`` (``False``).
    """
    if backend != "cake":
        raise ValueError(
            "the chunked LM-head + loss kernels currently support backend='cake'"
        )
    if check_num_valid_rows and num_valid_rows is None:
        raise ValueError(_knobs.CHECK_WITHOUT_COUNT)
    knobs = _knobs.resolve_knobs(
        compact_rows=compact_rows,
        fuse_dw_cast=fuse_dw_cast,
        dx_finalize=dx_finalize,
        dw_stream=dw_stream,
        hidden_count=hidden_count,
        gemm_tuning=gemm_tuning,
    )
    opt_in = memory_mode is not None or num_valid_rows is not None
    if not _no_temperature(temperature):
        if isinstance(temperature, bool):
            raise ValueError(
                "temperature must be None, a positive number or a float32 [T] tensor, not a bool"
            )
        if opt_in:
            temperature_entry = (
                _cake_backend_opt_in().cake_lm_head_nondet_train_logprob_temperature
            )
            return temperature_entry(
                X,
                W,
                labels,
                chunk_size=chunk_size,
                backend="cake",
                **knobs,
                memory_mode=memory_mode,
                num_valid_rows=num_valid_rows,
                check_num_valid_rows=check_num_valid_rows,
                temperature=temperature,
                check_temperature=check_temperature,
            )
        return _cake_backend().cake_lm_head_nondet_train_logprob_temperature(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            backend="cake",
            **knobs,
            temperature=temperature,
            check_temperature=check_temperature,
        )
    if opt_in:
        if hidden_count is None and memory_mode == _KEEP_LOGITS:
            # the kept logits keep the backend's kept-logits form for the count
            # (None: the form's default) unless the variable turns it off
            knobs["hidden_count"] = _kept_hidden_count()
        return _cake_backend_opt_in().cake_lm_head_nondet_train_logprob(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            backend="cake",
            **knobs,
            memory_mode=memory_mode,
            num_valid_rows=num_valid_rows,
            check_num_valid_rows=check_num_valid_rows,
        )
    return _cake_backend().cake_lm_head_nondet_train_logprob(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        backend="cake",
        **knobs,
    )


@flashinfer_api
def cake_lm_head_nondet_train_preload(
    device: Optional[Union[torch.device, int, str]] = None,
    *,
    temperature: bool = False,
):
    r"""Do part of the one-time work of a process's first chunked LM-head call
    ahead of it, then wait for the device once (SM100 / SM103 / SM107 only).

    See :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_preload`
    for the work; here it is done for this backend, whose libraries and
    per-device state the other entries use.  The per-thread state (side streams, events, count cells) is the
    calling thread's: another thread forms its own at its first call.
    ``temperature=True`` also loads the backend's temperature libraries and
    launches its temperature path's PyTorch operators once.
    """
    if temperature is False:
        _cake_backend_opt_in().cake_lm_head_nondet_train_preload(device)
        return
    _cake_backend_opt_in().cake_lm_head_nondet_train_preload(
        device, temperature=temperature
    )


@flashinfer_api
def cake_lm_head_nondet_train_support_logprob(
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
    r"""Differentiable log-probabilities over a ragged candidate support per
    token, SM100 / SM103 / SM107 only.

    See :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_logprob`
    for the computation and the arguments.
    The entry makes no run-to-run bitwise promise for the gradients: its
    backward raises PyTorch's alert under torch's deterministic mode before a
    dense dX of more than one K slice (with ``warn_only=True`` it warns and
    runs on); there is no fallback to the deterministic entry.
    """
    if backend != "cake":
        raise ValueError(
            "the candidate-support LM-head kernels currently support backend='cake'"
        )
    return _cake_support().cake_lm_head_nondet_train_support_logprob(
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
def cake_lm_head_nondet_train_support_kl(
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
    r"""Unreduced forward KL from a detached teacher over a ragged candidate
    support, SM100 / SM103 / SM107 only.

    See :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_kl`
    for the computation and the arguments; the gradients are as in
    :func:`cake_lm_head_nondet_train_support_logprob`.
    """
    if backend != "cake":
        raise ValueError(
            "the candidate-support LM-head kernels currently support backend='cake'"
        )
    return _cake_support().cake_lm_head_nondet_train_support_kl(
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
