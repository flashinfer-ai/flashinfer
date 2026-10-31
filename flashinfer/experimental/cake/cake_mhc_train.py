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

Cake four-stream mHC training forward / backward (4 streams; hidden 4096 on
SM100 / SM103 / SM107, and 5120 / 7168 on SM100 / SM103 with
``deterministic=True`` only) with a selectable numerical mode.  The function, the
tensor contracts, the workspaces and the autograd wiring are those of
:mod:`flashinfer.cake_mhc_det_train` (documented there).

Every entry point takes ``deterministic: bool = True``, except
:func:`cake_mhc_train_finish` (it follows its handle: the package whose
:func:`cake_mhc_train_prepare` made it), :func:`cake_mhc_train_reset_workspace`
and :func:`route_key` (the same in both packages):

* ``deterministic=True`` (the default) runs the deterministic package
  :mod:`flashinfer.cake_mhc_det_train` unchanged: the same programs and bitwise
  the same outputs as its ``cake_mhc_det_train_*`` names.
* ``deterministic=False`` runs the non-deterministic package
  ``flashinfer.experimental.cake.cake_mhc_train_backend``.  Its
  ``backward_finish`` has no final reduction kernel: the column-slab pass adds
  the ``dfn`` sum of every (column slab, token chunk) unit into ``dfn`` with
  FP32 reductions and folds ``dscale`` / ``dbase`` in the same grid.  Only the
  summation order of ``dfn`` varies from run to run: each element lies within
  ``P`` FP32 roundings of the fixed-order sum (``P <= 32`` token chunks), and
  the FP32 reduction flushes subnormal addends and results to zero.  ``dX``,
  ``dscale`` and ``dbase`` are bitwise the deterministic values, and so are the
  forward, the statistics replay and ``backward_prepare`` (copies of the
  deterministic programs).
* With ``torch.use_deterministic_algorithms(True)``, ``deterministic=False``
  runs the deterministic package and warns once per process
  (``UserWarning``).  The non-deterministic package itself never falls back:
  its ``backward_finish`` and the backward of its prepare node raise PyTorch's
  alert for an operation without a deterministic implementation (a
  ``RuntimeError``, or a ``UserWarning`` under ``warn_only=True``); so does the
  backward of a ``cake_mhc_train_prepare(..., deterministic=False)`` node when
  the deterministic algorithms are enabled between its forward and its
  backward.  Its forward, statistics, ``backward_prepare`` and workspace entry
  points never alert.

Workspaces: the forward workspace has one layout in both packages.  The
non-deterministic backward workspace is the deterministic one at the same
offsets without the per-chunk FP32 ``dfn`` partials (``P x 24 x 16384 x 4``
bytes), so a ``deterministic=True`` backward workspace serves both modes and a
``deterministic=False`` one serves the non-deterministic backward only.  When
torch's deterministic algorithms turn a ``deterministic=False`` call with a
caller-owned backward workspace too small for the deterministic layout into
the deterministic backward, that call runs on a private zero-initialised
companion buffer of the deterministic size, kept per caller workspace and
released with it (never the package's shared default buffer); inside a CUDA
graph capture a missing companion raises ``ValueError`` (pass a
``deterministic=True``-sized workspace, or run the call once before the
capture).  The outputs of either package feed the other's entry points (the
records are the same bits).

Default workspaces (``workspace=None``): each package keeps one buffer per
device and power-of-two token-count bucket, shared by every stream and every
graph replay on that device; calls that can run concurrently on different
streams pass caller-owned workspaces.  The non-deterministic package creates no
default buffer while the current stream of the buffer's device is capturing a
CUDA graph (``ValueError``): warm up with the same ``deterministic`` value and
token count before the capture.  A companion is allocated zeroed outside any
capture and every completed launch leaves its
ticket words at zero, so it needs no reset;
:func:`cake_mhc_train_reset_workspace` zeroes the buffer it is given.

Name map (the deterministic names are unchanged and always deterministic)::

    flashinfer.cake_mhc_det_train                 this module (deterministic=...)
      cake_mhc_det_train_forward_prepare            cake_mhc_train_forward_prepare
      cake_mhc_det_train_forward_stats              cake_mhc_train_forward_stats
      cake_mhc_det_train_forward_finish             cake_mhc_train_forward_finish
      cake_mhc_det_train_backward_prepare           cake_mhc_train_backward_prepare
      cake_mhc_det_train_backward_finish            cake_mhc_train_backward_finish
      cake_mhc_det_train_prepare                    cake_mhc_train_prepare
      cake_mhc_det_train_finish                     cake_mhc_train_finish (follows the handle)
      cake_mhc_det_train_{forward,backward}_        cake_mhc_train_{forward,backward}_
        workspace{,_bytes}                            workspace{,_bytes}
      cake_mhc_det_train_reset_workspace            cake_mhc_train_reset_workspace
      is_cake_mhc_det_train_supported               is_cake_mhc_train_supported
      CakeMHCDetTrainPrepareFunction                CakeMHCTrainPrepareFunction (non-deterministic)
      CakeMHCDetTrainFinishFunction                 CakeMHCTrainFinishFunction (non-deterministic package's)
      route_key                                     route_key

The legacy module ``flashinfer.cake_mhc_train`` (and its backend
``flashinfer.cake_mhc_train_backend``) keeps the same ``cake_mhc_train_*`` /
``CakeMHCTrain*`` names as aliases of the deterministic objects, without a
``deterministic`` keyword; this module is the explicit namespace of the
selectable mode.  The model's head (``cake_mhc_det_train_head*``) has no
cross-token reduction and no counterpart here: use
``flashinfer.cake_mhc_det_train.cake_mhc_det_train_head_forward`` /
``_head_backward`` / ``_head``.

Usage::

    from flashinfer.experimental.cake.cake_mhc_train import (
        cake_mhc_train_backward_finish,
        cake_mhc_train_backward_workspace,
        cake_mhc_train_finish,
        cake_mhc_train_prepare,
    )

    U, handle = cake_mhc_train_prepare(X, fn, scale, base, deterministic=False)
    Y = cake_mhc_train_finish(X, branch(U), handle)          # backward: atomic dfn

    ws_b = cake_mhc_train_backward_workspace(T, X.device, deterministic=False)
    dX, dfn, dscale, dbase = cake_mhc_train_backward_finish(
        X, dY, dU, B, fn, scale, base, stats, grad_stats,
        workspace=ws_b, deterministic=False)
"""

from __future__ import annotations

import functools
import warnings
import weakref
from typing import Any, Optional

import torch

from ... import cake_mhc_det_train_backend as _det
from ...api_logging import flashinfer_experimental_api
from ...cake_mhc_det_train_backend import cake_jit as _det_jit
from . import cake_mhc_train_backend as _nd
from .cake_mhc_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeMHCTrainFinishFunction,
    CakeMHCTrainPrepareFunction,
    cake_mhc_train_reset_workspace,
    route_key,
)
from .cake_mhc_train_backend import cake_jit as _nd_jit

# One warning per process when torch's deterministic algorithms turn a
# ``deterministic=False`` call deterministic.
_DET_FALLBACK_WARNED = False


_FEATURE = (
    "Cake four-stream mHC training with a selectable numerical mode (SM100/SM103/SM107)"
)


def _deterministic(deterministic: bool, entry: str) -> bool:
    """``True`` when ``entry`` runs the deterministic package: ``deterministic=True``, or ``deterministic=False`` while
    ``torch.are_deterministic_algorithms_enabled()`` (warned once per process)."""
    global _DET_FALLBACK_WARNED
    if not isinstance(deterministic, bool):
        raise TypeError(
            f"{entry}: deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if deterministic:
        return True
    if torch.are_deterministic_algorithms_enabled():
        if not _DET_FALLBACK_WARNED:
            _DET_FALLBACK_WARNED = True
            warnings.warn(
                f"{entry}: deterministic=False runs the deterministic package because torch's deterministic "
                "algorithms are enabled (torch.use_deterministic_algorithms(True)); this warning is shown once",
                UserWarning,
                stacklevel=3,
            )
        return True
    return False


# ----------------------------------------------------------------------------
# The backward workspace of a deterministic fallback
#
# Companion buffers per caller-owned backward workspace: ``id(workspace) ->
# (weak reference, [companions in ascending size])``.  A companion lives as long
# as the caller's workspace (the weak reference drops the entry with it), so a
# CUDA graph captured on the fallback path keeps a valid companion exactly as
# long as it keeps a valid caller workspace; companions are never replaced.
_COMPANIONS: dict[int, tuple[Any, list[torch.Tensor]]] = {}


def _forget_companions(key: int, ref: Any) -> None:
    entry = _COMPANIONS.get(key)
    if entry is not None and entry[0] is ref:
        del _COMPANIONS[key]


def _fallback_backward_workspace(workspace: Any, X: Any, entry: str) -> Any:
    """The backward workspace a deterministic fallback of a ``deterministic=False`` call runs on.

    ``None`` stays ``None`` (the deterministic package's default buffer, as
    for any deterministic call).  A workspace that holds the deterministic
    layout for ``X``'s token count passes unchanged, and so does anything that
    is not a valid ``deterministic=False`` workspace for that token count (a
    non-tensor, another device, dtype, rank or alignment, too few bytes, or an
    ``X`` the deterministic checks refuse): the deterministic package's checks
    refuse it in their own order, so an invalid workspace is never replaced.  A
    valid ``deterministic=False`` workspace too small for the deterministic
    layout runs on the companion of that workspace (see :data:`_COMPANIONS`): a
    zero-initialised buffer of the deterministic size of the token-count bucket,
    allocated on first need and reused by every later fallback on it.  Inside a
    CUDA graph capture a missing companion is a ``ValueError``.
    """

    if type(workspace) is not torch.Tensor:
        return workspace
    if not (
        isinstance(X, torch.Tensor)
        and X.is_cuda
        and X.dtype == torch.bfloat16
        and X.ndim == 3
        and tuple(X.shape[1:]) == (_det_jit.STREAMS, _det_jit.HIDDEN)
        and X.shape[0] > 0
    ):
        return workspace
    if (
        workspace.device != X.device
        or workspace.dtype != torch.uint8
        or workspace.ndim != 1
        or not workspace.is_contiguous()
        or workspace.data_ptr() % _nd.WORKSPACE_ALIGNMENT_BYTES != 0
    ):
        return workspace
    index = X.get_device()
    arch = _det_jit.arch_for_capability(torch.cuda.get_device_capability(index))
    if arch is None:
        return workspace
    rows = int(X.shape[0])
    sm_count = _det_jit.sm_count(index)
    det_need = _det_jit.workspace_bytes("bwd", rows, sm_count, arch)
    if workspace.numel() >= det_need:
        return workspace
    if workspace.numel() < _nd_jit.workspace_bytes("bwd", rows, sm_count, arch):
        return workspace
    key = id(workspace)
    found = _COMPANIONS.get(key)
    if found is None or found[0]() is not workspace:
        found = (weakref.ref(workspace, functools.partial(_forget_companions, key)), [])
        _COMPANIONS[key] = found
    for companion in found[1]:
        if companion.numel() >= det_need:
            return companion
    if torch.cuda.is_current_stream_capturing():
        raise ValueError(
            f"{entry}(deterministic=False) runs the deterministic backward because torch's deterministic "
            "algorithms are enabled, and the caller-owned backward workspace is too small for its layout "
            f"({workspace.numel()} bytes; T={rows} needs {det_need}); no companion buffer can be allocated "
            "during a CUDA graph capture: pass a workspace of "
            "cake_mhc_train_backward_workspace_bytes(T, device, deterministic=True) bytes (it serves both "
            "modes), or run the call once before the capture"
        )
    bucket = 1 << (rows - 1).bit_length()
    companion = torch.zeros(
        _det.cake_mhc_det_train_backward_workspace_bytes(bucket, workspace.device),
        dtype=torch.uint8,
        device=workspace.device,
    )
    found[1].append(companion)
    return companion


# ----------------------------------------------------------------------------
# Entry points (real ``def``s: public-API tooling reads the decorated
# signatures of this module; the bodies live in the two packages)


@flashinfer_experimental_api(feature=_FEATURE)
def is_cake_mhc_train_supported(
    device: torch.device,
    hidden: Optional[int] = None,
    *,
    deterministic: bool = True,
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has every exported route of the package ``deterministic``
    selects (the deterministic package's routes, or the non-deterministic package's)."""

    if _deterministic(deterministic, "is_cake_mhc_train_supported"):
        return _det.is_cake_mhc_det_train_supported(device, hidden)
    return _nd.is_cake_mhc_train_supported(device, hidden)


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_forward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = _det_jit.HIDDEN,
    deterministic: bool = True,
) -> int:
    """Bytes of the caller-owned forward workspace for up to ``T`` tokens on ``device``.

    The forward workspace has one layout for ``deterministic=True`` and ``deterministic=False`` (see
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_forward_workspace_bytes`).  ``hidden`` is the
    hidden size of the calls the buffer serves: 4096, 5120 or 7168 for ``deterministic=True``, 4096 for
    ``deterministic=False`` (any other value raises ``ValueError``).
    """

    if _deterministic(deterministic, "cake_mhc_train_forward_workspace_bytes"):
        return _det.cake_mhc_det_train_forward_workspace_bytes(T, device, hidden=hidden)
    return _nd.cake_mhc_train_forward_workspace_bytes(
        T, device, hidden=hidden, deterministic=False
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_backward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = _det_jit.HIDDEN,
    deterministic: bool = True,
) -> int:
    """Bytes of the caller-owned backward workspace for up to ``T`` tokens on ``device``.

    ``deterministic=True``: the deterministic package's workspace, which serves
    both backwards.  ``deterministic=False``: the non-deterministic backward's,
    the same regions at the same offsets without the per-chunk FP32 ``dfn``
    partials (``P x 24 x 16384 x 4`` bytes, ``P <= 32``).  The size is the
    maximum over every token count up to ``T``; ``T == 0`` needs no workspace.
    ``hidden`` as in :func:`cake_mhc_train_forward_workspace_bytes`.
    """

    if _deterministic(deterministic, "cake_mhc_train_backward_workspace_bytes"):
        return _det.cake_mhc_det_train_backward_workspace_bytes(
            T, device, hidden=hidden
        )
    return _nd.cake_mhc_train_backward_workspace_bytes(
        T, device, hidden=hidden, deterministic=False
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_forward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = _det_jit.HIDDEN,
    deterministic: bool = True,
) -> torch.Tensor:
    """Allocate a zero-initialized forward workspace for ``deterministic=True`` or ``deterministic=False`` (one layout;
    see :func:`cake_mhc_train_forward_workspace_bytes`)."""

    if _deterministic(deterministic, "cake_mhc_train_forward_workspace"):
        return _det.cake_mhc_det_train_forward_workspace(T, device, hidden=hidden)
    return _nd.cake_mhc_train_forward_workspace(
        T, device, hidden=hidden, deterministic=False
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_backward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = _det_jit.HIDDEN,
    deterministic: bool = True,
) -> torch.Tensor:
    """Allocate a zero-initialized backward workspace of the ``deterministic`` mode's layout (see
    :func:`cake_mhc_train_backward_workspace_bytes`)."""

    if _deterministic(deterministic, "cake_mhc_train_backward_workspace"):
        return _det.cake_mhc_det_train_backward_workspace(T, device, hidden=hidden)
    return _nd.cake_mhc_train_backward_workspace(
        T, device, hidden=hidden, deterministic=False
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_forward_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    out: Optional[torch.Tensor] = None,
    stats_out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Forward mapping, coefficients and collapse: ``(U, stats)``.

    The arguments and outputs are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_forward_prepare`.
    ``deterministic`` selects the package (see the module docstring); both run
    the same programs, so ``U`` and ``stats`` are the same bits.
    """

    if _deterministic(deterministic, "cake_mhc_train_forward_prepare"):
        return _det.cake_mhc_det_train_forward_prepare(
            X,
            fn,
            scale,
            base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            out=out,
            stats_out=stats_out,
            workspace=workspace,
        )
    return _nd.cake_mhc_train_forward_prepare(
        X,
        fn,
        scale,
        base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        out=out,
        stats_out=stats_out,
        workspace=workspace,
        deterministic=False,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_forward_stats(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    stats_out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    r"""Recompute the per-token statistics of ``forward_prepare`` (checkpoint replay).

    The arguments and output are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_forward_stats`;
    ``deterministic`` selects the package (the same bits in both).
    """

    if _deterministic(deterministic, "cake_mhc_train_forward_stats"):
        return _det.cake_mhc_det_train_forward_stats(
            X,
            fn,
            scale,
            base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            stats_out=stats_out,
            workspace=workspace,
        )
    return _nd.cake_mhc_train_forward_stats(
        X,
        fn,
        scale,
        base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        stats_out=stats_out,
        workspace=workspace,
        deterministic=False,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_forward_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    stats: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    r"""Residual mixing: ``Y[t, j, d] = BF16(post[t, j] B[t, d] + sum_i C[t, i, j] X[t, i, d])``.

    The arguments and output are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_forward_finish`;
    ``deterministic`` selects the package (the same bits in both).
    """

    if _deterministic(deterministic, "cake_mhc_train_forward_finish"):
        return _det.cake_mhc_det_train_forward_finish(X, B, stats, out=out)
    return _nd.cake_mhc_train_forward_finish(X, B, stats, out=out, deterministic=False)


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_backward_prepare(
    dY: torch.Tensor,
    B: torch.Tensor,
    X: torch.Tensor,
    stats: torch.Tensor,
    *,
    dB: Optional[torch.Tensor] = None,
    grad_stats_out: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Branch-output gradient and the per-token gradient statistics: ``(dB, grad_stats)``.

    The arguments and outputs are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_prepare`;
    ``deterministic`` selects the package (the same bits in both: this pass has
    no cross-token reduction).
    """

    if _deterministic(deterministic, "cake_mhc_train_backward_prepare"):
        return _det.cake_mhc_det_train_backward_prepare(
            dY, B, X, stats, dB=dB, grad_stats_out=grad_stats_out
        )
    return _nd.cake_mhc_train_backward_prepare(
        dY, B, X, stats, dB=dB, grad_stats_out=grad_stats_out, deterministic=False
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_backward_finish(
    X: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    B: Optional[torch.Tensor],
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    grad_stats: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulate: bool = False,
    workspace: Optional[torch.Tensor] = None,
    dX: Optional[torch.Tensor] = None,
    dfn: Optional[torch.Tensor] = None,
    dscale: Optional[torch.Tensor] = None,
    dbase: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Input gradient and parameter gradients: ``(dX, dfn, dscale, dbase)``.

    The arguments and outputs are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_finish`.

    Parameters
    ----------
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_train_backward_workspace`
        (``deterministic=True`` sizes one for both backwards, ``False`` for the
        non-deterministic backward only); a per-device default when omitted.
    deterministic : bool
        ``True`` (default): the deterministic backward, bitwise
        :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_finish`.
        ``False``: the non-deterministic backward (``dfn`` summed with FP32
        reductions in a run-to-run varying order, each element within
        ``P <= 32`` FP32 roundings of the fixed-order sum; ``dX``, ``dscale``,
        ``dbase`` bitwise the deterministic values; ``accumulate=True`` adds onto
        the caller's values the same way).  Under
        ``torch.use_deterministic_algorithms(True)`` a ``False`` call runs the
        deterministic backward (one warning per process); a caller workspace
        too small for its layout is then replaced by a private companion buffer
        (see the module docstring).
    """

    if _deterministic(deterministic, "cake_mhc_train_backward_finish"):
        if deterministic is False:
            workspace = _fallback_backward_workspace(
                workspace, X, "cake_mhc_train_backward_finish"
            )
        return _det.cake_mhc_det_train_backward_finish(
            X,
            dY,
            dU,
            B,
            fn,
            scale,
            base,
            stats,
            grad_stats,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            accumulate=accumulate,
            workspace=workspace,
            dX=dX,
            dfn=dfn,
            dscale=dscale,
            dbase=dbase,
        )
    return _nd.cake_mhc_train_backward_finish(
        X,
        dY,
        dU,
        B,
        fn,
        scale,
        base,
        stats,
        grad_stats,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulate=accumulate,
        workspace=workspace,
        dX=dX,
        dfn=dfn,
        dscale=dscale,
        dbase=dbase,
        deterministic=False,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    save_stats: bool = True,
    workspace: Optional[torch.Tensor] = None,
    backward_workspace: Optional[torch.Tensor] = None,
    deterministic: bool = True,
):
    """Autograd-enabled ``forward_prepare``: ``(U, handle)``; pass ``handle`` to :func:`cake_mhc_train_finish`.

    The arguments and outputs are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_prepare`.

    Parameters
    ----------
    backward_workspace : torch.Tensor, optional
        Caller-owned backward workspace of :func:`cake_mhc_train_backward_workspace`
        with the same ``deterministic`` (or ``True``, which serves both); a
        per-device default when omitted.
    deterministic : bool
        ``True`` (default): the deterministic package's prepare node, its
        gradients bit for bit.  ``False``: the non-deterministic package's node,
        whose backward runs the non-deterministic ``backward_finish`` (see
        :func:`cake_mhc_train_backward_finish`); the forward and its checkpoint
        replay are the same bits.  Under
        ``torch.use_deterministic_algorithms(True)`` when this is called, a
        ``False`` call makes the deterministic node (one warning per process; a
        ``backward_workspace`` too small for the deterministic layout is
        replaced by its companion buffer, see the module docstring).  A
        ``False`` node whose backward runs after the deterministic algorithms
        were enabled raises PyTorch's alert (``RuntimeError``; under
        ``warn_only=True`` a ``UserWarning``, and it runs).
    """

    if _deterministic(deterministic, "cake_mhc_train_prepare"):
        if deterministic is False:
            backward_workspace = _fallback_backward_workspace(
                backward_workspace, X, "cake_mhc_train_prepare"
            )
        return _det.cake_mhc_det_train_prepare(
            X,
            fn,
            scale,
            base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            save_stats=save_stats,
            workspace=workspace,
            backward_workspace=backward_workspace,
        )
    return _nd.cake_mhc_train_prepare(
        X,
        fn,
        scale,
        base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        save_stats=save_stats,
        workspace=workspace,
        backward_workspace=backward_workspace,
        deterministic=False,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_finish(X: torch.Tensor, B: torch.Tensor, handle) -> torch.Tensor:
    """Autograd-enabled ``forward_finish``: ``Y`` from ``X``, the branch output ``B`` and the prepare ``handle``.

    The arguments and output are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_finish`.  The
    handle names the package: the finish node of the package whose prepare node
    made it (no ``deterministic`` keyword).
    """

    if isinstance(handle, _nd._MHCTrainHandle):
        return _nd.cake_mhc_train_finish(X, B, handle, deterministic=False)
    if isinstance(handle, _det._MHCTrainHandle):
        return _det.cake_mhc_det_train_finish(X, B, handle)
    raise TypeError("handle must be the handle returned by cake_mhc_train_prepare")


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeMHCTrainFinishFunction",
    "CakeMHCTrainPrepareFunction",
    "cake_mhc_train_backward_finish",
    "cake_mhc_train_backward_prepare",
    "cake_mhc_train_backward_workspace",
    "cake_mhc_train_backward_workspace_bytes",
    "cake_mhc_train_finish",
    "cake_mhc_train_forward_finish",
    "cake_mhc_train_forward_prepare",
    "cake_mhc_train_forward_stats",
    "cake_mhc_train_forward_workspace",
    "cake_mhc_train_forward_workspace_bytes",
    "cake_mhc_train_prepare",
    "cake_mhc_train_reset_workspace",
    "is_cake_mhc_train_supported",
    "route_key",
]
