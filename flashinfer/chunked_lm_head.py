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
import struct
from types import ModuleType
from typing import Any, Dict, NoReturn, Optional, Union

import torch

from . import _cake_lm_head_train_knobs as _knobs
from .api_logging import flashinfer_experimental_api

# Thin experimental entry points of the non-deterministic LM-head package,
# flashinfer.experimental.cake.cake_lm_head_loss: the det-free entries
# ``chunked_lm_head_{loss,logprob,preload,support_logprob,support_kl}`` run that
# package only.  They need ``deterministic=False``; any other value, the
# default included, is a ``ValueError`` naming the deterministic entry in
# ``flashinfer.cake_lm_head_det_train`` (see the section below).
#
# The FLASHINFER_CAKE_LM_HEAD_LOSS_* variables are the A/B knobs of
# ``flashinfer/_cake_lm_head_train_knobs.py``, shared with
# ``flashinfer.cake_lm_head_det_train``: a switch keyword left at ``None``
# falls back to its variable, and the resolved value is passed down.  The two
# binding-cache variables are read once, when this module is imported, and
# applied to the package's cache on first use.

_FEATURE = "chunked LM-head + loss training kernels (SM100/SM103/SM107)"

# Read once at import: a malformed value fails here, naming the variable, not
# on the first call.  Applied to the package's cache by ``_nd_backend``.
_BINDING_CACHE_SETTINGS: Dict[str, Any] = _knobs.binding_cache_settings()


# ---------------------------------------------------------------------------
# The det-free entries: the non-deterministic package only.
#
# ``chunked_lm_head_{loss,logprob,preload,support_logprob,support_kl}`` run
# the non-deterministic package, flashinfer.experimental.cake.cake_lm_head_loss,
# and need ``deterministic=False``.  Any other value, the default ``True``
# included, raises ``ValueError`` naming the deterministic entry to call
# instead (``flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_<same
# suffix>``), before any package import or
# torch query.  ``deterministic=False`` runs that package under the same
# argument checks and the same FLASHINFER_CAKE_LM_HEAD_LOSS_* fallbacks, also
# while torch's deterministic algorithms are enabled
# (``torch.use_deterministic_algorithms(True)``, also with ``warn_only``): the
# only alert is PyTorch's, raised by that package before a K-sliced dX launch
# (docs/training_backend_names.md); there is no fallback to the deterministic
# entry, every knob (a dX ``k_slices`` pin included) reaches the package
# unchanged and these entries warn nothing themselves.  Each entry's
# signature (``__signature__``) is its mode-free helper's below plus the
# keyword-only ``deterministic: bool = True``.
# ---------------------------------------------------------------------------


def _require_nd(deterministic: object, entry: str, det_entry: str) -> None:
    """Refuse every ``deterministic`` but ``False`` for the det-free ``entry``,
    naming ``det_entry``, the deterministic entry to call instead."""
    if deterministic is not False:
        raise ValueError(
            f"{entry}() runs the non-deterministic package "
            "flashinfer.experimental.cake.cake_lm_head_loss only; pass "
            "deterministic=False for the non-deterministic path (got "
            f"deterministic={deterministic!r}); the deterministic entry is "
            f"flashinfer.cake_lm_head_det_train.{det_entry}"
        )


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


@functools.cache
def _nd_backend() -> ModuleType:
    # The non-deterministic package, imported on its first use (a call with
    # ``deterministic=False``); its binding cache takes the same settings.
    from .experimental.cake.cake_lm_head_loss import cake_backend

    _knobs.configure_binding_cache(cake_backend, _BINDING_CACHE_SETTINGS)
    return cake_backend


@functools.cache
def _nd_backend_opt_in() -> ModuleType:
    _nd_backend()
    from .experimental.cake.cake_lm_head_loss import cake_backend_opt_in

    _knobs.configure_binding_cache(cake_backend_opt_in, _BINDING_CACHE_SETTINGS)
    return cake_backend_opt_in


@functools.cache
def _nd_support() -> ModuleType:
    _nd_backend()
    from .experimental.cake.cake_lm_head_loss import cake_support

    return cake_support


def _no_temperature(temperature: object) -> bool:
    """Whether ``temperature`` is no temperature for the non-deterministic entry (which then runs the call without
    one): ``None`` or a Python number (not a ``bool``) whose FP32 value is 1.0."""
    if temperature is None:
        return True
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        return False
    try:
        return struct.unpack("<f", struct.pack("<f", temperature))[0] == 1.0
    except OverflowError:
        return False


def _with_mode(fn, *, before: str) -> inspect.Signature:
    """``fn``'s signature with the keyword-only ``deterministic: bool = True``
    inserted before the parameter ``before``: the displayed signature of a
    det-free entry, formed from its mode-free helper ``fn``."""
    params = list(inspect.signature(fn).parameters.values())
    at = [p.name for p in params].index(before)
    mode = inspect.Parameter(
        "deterministic", inspect.Parameter.KEYWORD_ONLY, default=True, annotation=bool
    )
    return inspect.signature(fn).replace(parameters=params[:at] + [mode] + params[at:])


def _nd_loss(
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
    """The body of :func:`chunked_lm_head_loss` after its mode check."""
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
    r"""Differentiable chunked LM-head projection + loss through the
    non-deterministic package, SM100 / SM103 / SM107 only.

    The computation, the arguments and the outputs are those of
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss` (each switch keyword left at ``None``
    reads the same ``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` variable).

    Parameters
    ----------
    deterministic : bool
        Only ``False`` is accepted: the non-deterministic package
        (``flashinfer.experimental.cake.cake_lm_head_loss``): the same
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
        Any other value, the default ``True`` included, raises ``ValueError``
        naming :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss`, the deterministic entry to
        call instead (it takes no ``deterministic`` keyword).  While
        ``torch.use_deterministic_algorithms(True)`` is in effect the call
        still runs the non-deterministic package: PyTorch's alert is raised
        by that package before a K-sliced dX launch (``RuntimeError``; with
        ``warn_only=True`` a ``UserWarning``, then the call runs); there is
        no fallback to the deterministic entry, and every knob reaches the
        package unchanged.

    Every other parameter, the return value and the notes: see
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss`.
    """
    _require_nd(deterministic, "chunked_lm_head_loss", "cake_lm_head_det_train_loss")
    try:
        return _nd_loss(*args, **kwargs)
    except TypeError as exc:
        _raise_named(exc, _nd_loss, "chunked_lm_head_loss")


chunked_lm_head_loss.__wrapped__.__signature__ = _with_mode(_nd_loss, before="backend")


def _nd_logprob(
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
    """The body of :func:`chunked_lm_head_logprob` after its mode check."""
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
    if not _no_temperature(temperature):
        # the backends' temperature entries (as the deterministic entry routes a temperature); ``None`` and a number
        # whose FP32 value is 1.0 are the call without one
        if isinstance(temperature, bool):
            raise ValueError(
                "temperature must be None, a positive number or a float32 [T] tensor, not a bool"
            )
        if memory_mode is not None or num_valid_rows is not None:
            return _nd_backend_opt_in().chunked_lm_head_logprob_temperature(
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
                temperature=temperature,
                check_temperature=check_temperature,
            )
        return _nd_backend().chunked_lm_head_logprob_temperature(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            deterministic=False,
            backend="cake",
            **knobs,
            temperature=temperature,
            check_temperature=check_temperature,
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
    r"""Differentiable chunked selected-token log-probability through the
    non-deterministic package, SM100 / SM103 / SM107 only.

    The computation and the arguments are those of
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_logprob`.  ``deterministic=False`` (the only
    accepted value): the non-deterministic package
    (``flashinfer.experimental.cake.cake_lm_head_loss``), whose backward adds
    each K slice of a sliced dX GEMM into the FP32 ``dX`` accumulator by an
    unordered FP32 reduction (``logp`` keeps its fixed order).  Any other
    value, the default ``True`` included, raises ``ValueError`` naming
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_logprob`, the deterministic entry to call
    instead.  Under ``torch.use_deterministic_algorithms(True)`` the call
    still runs the non-deterministic package: PyTorch's alert is raised by
    that package before a K-sliced dX launch; there is no fallback to the
    deterministic entry.
    ``temperature`` / ``check_temperature``: the non-deterministic package
    runs the same temperature forms (``logp`` the deterministic entry's bits,
    ``dW`` too when no valid row has a zero ``dlogp``; ``dX`` the same
    products, a sliced chunk's in the unordered reduction) and its
    zero-gradient row skip runs the dX / dW GEMMs of the rows with a nonzero
    ``dlogp`` only, with a temperature too.
    """
    _require_nd(
        deterministic, "chunked_lm_head_logprob", "cake_lm_head_det_train_logprob"
    )
    try:
        return _nd_logprob(*args, **kwargs)
    except TypeError as exc:
        _raise_named(exc, _nd_logprob, "chunked_lm_head_logprob")


chunked_lm_head_logprob.__wrapped__.__signature__ = _with_mode(
    _nd_logprob, before="backend"
)


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_preload(
    device: Optional[Union[torch.device, int, str]] = None,
    *,
    deterministic: bool = True,
    temperature: bool = False,
):
    r"""Do part of the one-time work of a process's first chunked LM-head call
    ahead of it, then wait for the device once (SM100 / SM103 / SM107 only).

    ``deterministic=False`` (the only accepted value): the work of
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_preload` for the non-deterministic package
    (``flashinfer.experimental.cake.cake_lm_head_loss``), whose libraries and
    per-device state the other det-free entries use.  Any other value, the
    default ``True`` included, raises ``ValueError`` naming
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_preload`, the deterministic package's preload.
    The per-thread state (side streams, events, count cells) is the calling
    thread's: another thread forms its own at its first call.
    ``temperature=True`` also loads the package's temperature libraries and
    launches its temperature path's PyTorch operators once.
    """
    _require_nd(
        deterministic, "chunked_lm_head_preload", "cake_lm_head_det_train_preload"
    )
    if temperature is False:
        _nd_backend_opt_in().preload(device)
        return
    _nd_backend_opt_in().preload(device, temperature=temperature)


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
    backend: str = "cake",
) -> torch.Tensor:
    """The body of :func:`chunked_lm_head_support_logprob` after its mode check."""
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
    token through the non-deterministic package, SM100 / SM103 / SM107 only.

    The computation and the other arguments are those of
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_logprob`.  ``deterministic=False`` (the
    only accepted value): the non-deterministic package's entry
    (``flashinfer.experimental.cake.cake_lm_head_loss.cake_support``), which
    makes no run-to-run bitwise promise for the gradients.  Any other value,
    the default ``True`` included, raises ``ValueError`` naming
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_logprob`, the deterministic entry to
    call instead.  Under ``torch.use_deterministic_algorithms(True)`` the call
    still runs the non-deterministic package: PyTorch's alert is raised by
    that package before a K-sliced dX launch; there is no fallback to the
    deterministic entry.
    """
    _require_nd(
        deterministic,
        "chunked_lm_head_support_logprob",
        "cake_lm_head_det_train_support_logprob",
    )
    try:
        return _nd_support_logprob(*args, **kwargs)
    except TypeError as exc:
        _raise_named(exc, _nd_support_logprob, "chunked_lm_head_support_logprob")


chunked_lm_head_support_logprob.__wrapped__.__signature__ = _with_mode(
    _nd_support_logprob, before="backend"
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
    backend: str = "cake",
):
    """The body of :func:`chunked_lm_head_support_kl` after its mode check."""
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
    support through the non-deterministic package, SM100 / SM103 / SM107 only.

    The computation and the other arguments are those of
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_kl`; ``deterministic`` as in
    :func:`chunked_lm_head_support_logprob` (any value but ``False``, the
    default ``True`` included, raises ``ValueError`` naming
    :func:`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_kl`).
    """
    _require_nd(
        deterministic, "chunked_lm_head_support_kl", "cake_lm_head_det_train_support_kl"
    )
    try:
        return _nd_support_kl(*args, **kwargs)
    except TypeError as exc:
        _raise_named(exc, _nd_support_kl, "chunked_lm_head_support_kl")


chunked_lm_head_support_kl.__wrapped__.__signature__ = _with_mode(
    _nd_support_kl, before="backend"
)
