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

Cake four-stream mHC training forward / backward, ``nondet`` mode
(4 streams; hidden 4096 on SM100 / SM103 / SM107, and 5120 / 7168 on SM100 /
SM103; another hidden size raises ``ValueError``): every entry point runs the
package ``flashinfer.experimental.cake.cake_mhc_nondet_train``, imported on
the first call or attribute access (importing this module imports no
backend).  The function, the tensor contracts and the autograd wiring are
those of the deterministic module :mod:`flashinfer.cake_mhc_det_train`
(documented there); the workspaces are described below.

Mode: the mode is part of the names (``cake_mhc_nondet_train_*``,
``CakeMHCNondet*Function``).  No entry point takes a ``deterministic``
keyword (passing one is a ``TypeError``), and nothing falls back to the
deterministic package ``flashinfer.cake_mhc_det_train``.

Numerical contract: every output repeats bit for bit from run to run on one
device (each sum runs in an order fixed by the token count and the device).
``backward_finish`` has no final reduction kernel: the column-slab pass
stores the compensated ``dfn`` sum of every (column slab, token chunk) unit,
the deterministic package's FP32 partial of that unit bit for bit, in its
chunk's slots of the backward workspace, and once every unit has stored its
sums the grid folds the slots (``P <= 32`` token chunks, the deterministic
package's chunks per token count and device) into ``dfn`` in ascending chunk
order as a compensated FP32 sum, the caller's ``dfn`` last under
``accumulate=True``, with one rounding: within one FP32 rounding of the exact
sum plus ``O(P^2 u^2) (|start| + sum |x_p|)`` per element (``x_p`` the
partials, ``u = 2^-24``).  The deterministic package sums the same partials
in FP64 and rounds once, so ``dfn`` is the deterministic ``dfn`` except where
the exact sum lies within that last term of an FP32 rounding boundary.  ``dX``, ``dscale`` and ``dbase`` are
bitwise the deterministic values (the column-slab pass folds the parameter
gradients in the deterministic reduction's order), and so are the forward,
the statistics replay and ``backward_prepare`` (copies of the deterministic
programs).  Under ``torch.use_deterministic_algorithms(True)`` no entry point
raises PyTorch's alert or warning.

Workspaces: the forward workspace has one layout in both packages.  The
backward workspace has this package's own layout: a 256-byte ticket region
and 512 bytes of column-slab arrival counts first (zero on first use), the
deterministic package's regions without its per-chunk FP32 ``dfn`` partials,
then the column slab's per-chunk FP32 ``dfn`` slots
(``min(32, ceil(T / 64)) x 24 x 4H x 4`` bytes at hidden size ``H``).  A
deterministic backward workspace does not serve this backward: a buffer the
deterministic package's backward used must be zeroed
(:func:`cake_mhc_nondet_train_reset_workspace`) before this package uses it.
The records of either package (``stats``, ``grad_stats``) feed the other's
entry points (the same bits); a prepare handle feeds only the finish of its
own package (``TypeError`` otherwise).

Default workspaces (``workspace=None``): one buffer per device, hidden size and
power-of-two token-count bucket, shared by every stream and every graph replay
on that device; calls that can run concurrently on different streams pass
caller-owned workspaces.  No default buffer is created while the current stream
of the buffer's device is capturing a CUDA graph (``ValueError``): warm up with
the same token count and hidden size before the capture.  Every completed
launch leaves the ticket words and the arrival counts of its workspace at
zero, so a replay needs no reset; :func:`cake_mhc_nondet_train_reset_workspace`
zeroes the buffer it is given.

Name map (the same verbs in both modes)::

    flashinfer.cake_mhc_det_train                 flashinfer.cake_mhc_nondet_train
      cake_mhc_det_train_forward_prepare            cake_mhc_nondet_train_forward_prepare
      cake_mhc_det_train_forward_stats              cake_mhc_nondet_train_forward_stats
      cake_mhc_det_train_forward_finish             cake_mhc_nondet_train_forward_finish
      cake_mhc_det_train_backward_prepare           cake_mhc_nondet_train_backward_prepare
      cake_mhc_det_train_backward_finish            cake_mhc_nondet_train_backward_finish
      cake_mhc_det_train_prepare                    cake_mhc_nondet_train_prepare
      cake_mhc_det_train_finish                     cake_mhc_nondet_train_finish
      cake_mhc_det_train_{forward,backward}_        cake_mhc_nondet_train_{forward,backward}_
        workspace{,_bytes}                            workspace{,_bytes}
      cake_mhc_det_train_reset_workspace            cake_mhc_nondet_train_reset_workspace
      is_cake_mhc_det_train_supported               is_cake_mhc_nondet_train_supported
      CakeMHCDetPrepareFunction                     CakeMHCNondetPrepareFunction
      CakeMHCDetFinishFunction                      CakeMHCNondetFinishFunction
      cake_mhc_det_train_route_key                  cake_mhc_nondet_train_route_key

The model's head is a module of its own in this mode:
``flashinfer.cake_mhc_head_nondet_train`` (``cake_mhc_head_nondet_train_forward``
/ ``cake_mhc_head_nondet_train_backward`` / ``cake_mhc_head_nondet``).

Usage::

    from flashinfer.cake_mhc_nondet_train import (
        cake_mhc_nondet_train_backward_finish,
        cake_mhc_nondet_train_backward_workspace,
        cake_mhc_nondet_train_finish,
        cake_mhc_nondet_train_prepare,
    )

    U, handle = cake_mhc_nondet_train_prepare(X, fn, scale, base)
    Y = cake_mhc_nondet_train_finish(X, branch(U), handle)  # backward: folded dfn

    ws_b = cake_mhc_nondet_train_backward_workspace(T, X.device, hidden=X.shape[2])
    dX, dfn, dscale, dbase = cake_mhc_nondet_train_backward_finish(
        X, dY, dU, B, fn, scale, base, stats, grad_stats, workspace=ws_b)
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api

# The default hidden size of the workspace helpers: the backend's ``cake_jit.HIDDEN``,
# spelled here so that defining the entry points imports no backend (a test checks
# that the two agree).
_HIDDEN = 4096


# The backend package is imported on first use and cached: a call does not repeat the import.
@functools.cache
def _pkg():
    from .experimental.cake import cake_mhc_nondet_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_mhc_nondet_train


# ----------------------------------------------------------------------------
# Entry points (real ``def``s: public-API tooling reads the decorated
# signatures of this module; the bodies live in the backend package)


@flashinfer_api
def is_cake_mhc_nondet_train_supported(
    device: torch.device,
    hidden: Optional[int] = None,
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has every exported route of the backend package.

    Parameters
    ----------
    device : torch.device
        Device to query; a device that is not CUDA, or whose architecture is not
        SM100 / SM103 / SM107, gives ``False``.
    hidden : int, optional
        Hidden size to query: 4096, 5120 or 7168 on SM100 / SM103, 4096 on
        SM107.  ``None`` (default) queries the architecture only.

    Returns
    -------
    ``True`` when the device (at ``hidden``, when given) has the routes.
    """

    return _pkg().is_cake_mhc_nondet_train_supported(device, hidden)


@flashinfer_api
def cake_mhc_nondet_train_forward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = _HIDDEN,
) -> int:
    """Bytes of the caller-owned forward workspace for up to ``T`` tokens on ``device``.

    The forward workspace has one layout in both packages (see
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_forward_workspace_bytes`).

    Parameters
    ----------
    T : int
        Largest token count the buffer serves (the size is the maximum over
        every token count up to ``T``); ``T == 0`` needs no workspace.
    device : torch.device
        CUDA device of the calls.
    hidden : int
        Hidden size ``H`` of the calls the buffer serves (``X.shape[2]``;
        default ``4096``): 4096, 5120 or 7168 on SM100 / SM103, 4096 on SM107
        (any other value raises ``ValueError``).

    Returns
    -------
    The size of the forward workspace in bytes.
    """

    return _pkg().cake_mhc_nondet_train_forward_workspace_bytes(
        T, device, hidden=hidden
    )


@flashinfer_api
def cake_mhc_nondet_train_backward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = _HIDDEN,
) -> int:
    """Bytes of the caller-owned backward workspace for up to ``T`` tokens on ``device``.

    This package's backward layout: a 256-byte ticket region and 512 bytes of
    column-slab arrival counts first (zero on first use; every completed
    launch leaves them zero), the deterministic package's regions without its
    per-chunk FP32 ``dfn`` partials, then the column slab's per-chunk FP32
    ``dfn`` slots (``min(32, ceil(T / 64)) x 24 x 4H x 4`` bytes at hidden
    size ``H``).  A deterministic backward workspace
    (:func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_workspace`)
    does not serve this backward.

    Parameters
    ----------
    T : int
        Largest token count the buffer serves (the size is the maximum over
        every token count up to ``T``); ``T == 0`` needs no workspace.
    device : torch.device
        CUDA device of the calls.
    hidden : int
        Hidden size of the calls the buffer serves (default ``4096``), as in
        :func:`cake_mhc_nondet_train_forward_workspace_bytes`.

    Returns
    -------
    The size of the backward workspace in bytes.
    """

    return _pkg().cake_mhc_nondet_train_backward_workspace_bytes(
        T, device, hidden=hidden
    )


@flashinfer_api
def cake_mhc_nondet_train_forward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = _HIDDEN,
) -> torch.Tensor:
    """Allocate a zero-initialized forward workspace (one layout in both packages; see
    :func:`cake_mhc_nondet_train_forward_workspace_bytes`).

    Parameters
    ----------
    T, device, hidden : int, torch.device, int
        As in :func:`cake_mhc_nondet_train_forward_workspace_bytes`.

    Returns
    -------
    A zero-filled ``uint8`` tensor of that many bytes on ``device``.
    """

    return _pkg().cake_mhc_nondet_train_forward_workspace(T, device, hidden=hidden)


@flashinfer_api
def cake_mhc_nondet_train_backward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = _HIDDEN,
) -> torch.Tensor:
    """Allocate a zero-initialized backward workspace of this package's layout (see
    :func:`cake_mhc_nondet_train_backward_workspace_bytes`).

    Parameters
    ----------
    T, device, hidden : int, torch.device, int
        As in :func:`cake_mhc_nondet_train_backward_workspace_bytes`.

    Returns
    -------
    A zero-filled ``uint8`` tensor of that many bytes on ``device``.
    """

    return _pkg().cake_mhc_nondet_train_backward_workspace(T, device, hidden=hidden)


@flashinfer_api
def cake_mhc_nondet_train_forward_prepare(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Forward mapping, coefficients and collapse: ``(U, stats)``.

    The arguments and outputs are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_forward_prepare`.
    The call runs the backend package, whose forward runs copies of the
    deterministic programs, so ``U`` and ``stats`` are the deterministic
    package's bits.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams; ``H = X.shape[2]`` is
        4096, 5120 or 7168 on SM100 / SM103 and 4096 on SM107.
    fn, scale, base : torch.Tensor
        FP32 ``[24, 4H]`` mapping matrix, ``[3]`` group scales and ``[24]``
        logit offsets.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        Normalization epsilon (default ``1e-5``), coefficient epsilon (default
        ``1e-6``) and Sinkhorn iteration count (1 to 64, default ``20``).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous ``U``.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned buffer from :func:`cake_mhc_nondet_train_forward_workspace`
        (``hidden=X.shape[2]``); a per-device default when omitted.

    Returns
    -------
    ``(U, stats)``: BF16 ``U`` of shape ``[T, H]`` and the FP32 ``[T, 64]``
    per-token record.
    """

    return _pkg().cake_mhc_nondet_train_forward_prepare(
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


@flashinfer_api
def cake_mhc_nondet_train_forward_stats(
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
) -> torch.Tensor:
    r"""Recompute the per-token statistics of ``forward_prepare`` (checkpoint replay).

    The arguments and output are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_forward_stats`;
    the call runs the backend package (the deterministic package's
    bits).

    Parameters
    ----------
    X, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (``hidden=X.shape[2]``); a per-device
        default when omitted.

    Returns
    -------
    FP32 ``[T, 64]`` ``stats``, bitwise the record of
    :func:`cake_mhc_nondet_train_forward_prepare` for the same inputs.
    """

    return _pkg().cake_mhc_nondet_train_forward_stats(
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


@flashinfer_api
def cake_mhc_nondet_train_forward_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    stats: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Residual mixing: ``Y[t, j, d] = BF16(post[t, j] B[t, d] + sum_i C[t, i, j] X[t, i, d])``.

    The arguments and output are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_forward_finish`;
    the call runs the backend package (the deterministic package's
    bits).

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams (the ``forward_prepare`` input).
    B : torch.Tensor
        BF16 ``[T, H]`` contiguous output of the external branch applied to ``U``.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of :func:`cake_mhc_nondet_train_forward_prepare`
        (or :func:`cake_mhc_nondet_train_forward_stats`).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous ``Y``.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, H]``.
    """

    return _pkg().cake_mhc_nondet_train_forward_finish(X, B, stats, out=out)


@flashinfer_api
def cake_mhc_nondet_train_backward_prepare(
    dY: torch.Tensor,
    B: torch.Tensor,
    X: torch.Tensor,
    stats: torch.Tensor,
    *,
    dB: Optional[torch.Tensor] = None,
    grad_stats_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Branch-output gradient and the per-token gradient statistics: ``(dB, grad_stats)``.

    The arguments and outputs are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_prepare`;
    the call runs the backend package (the deterministic package's
    bits: this pass has no cross-token reduction).

    Parameters
    ----------
    dY : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous gradient of ``Y``.
    B : torch.Tensor
        BF16 ``[T, H]`` contiguous branch output of the forward.
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams of the forward.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward.
    dB : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous output.
    grad_stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 32]`` contiguous output.

    Returns
    -------
    ``(dB, grad_stats)``: BF16 ``dB`` of shape ``[T, H]`` and the FP32
    ``[T, 32]`` record that :func:`cake_mhc_nondet_train_backward_finish` consumes.
    """

    return _pkg().cake_mhc_nondet_train_backward_prepare(
        dY, B, X, stats, dB=dB, grad_stats_out=grad_stats_out
    )


@flashinfer_api
def cake_mhc_nondet_train_backward_finish(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Input gradient and parameter gradients: ``(dX, dfn, dscale, dbase)``.

    The arguments and outputs are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_finish`;
    the call runs the backend package's backward, with no final reduction
    kernel.  Every output repeats bit for bit from run to run on one device.
    ``dX``, ``dscale`` and ``dbase`` are bitwise the deterministic values.
    Each element of ``dfn`` folds its ``P <= 32`` per-chunk FP32 partials
    ``x_p`` (bitwise the deterministic package's) in ascending chunk order as
    a compensated FP32 sum, the caller's ``dfn`` last under
    ``accumulate=True``, with one rounding: within one FP32 rounding of the
    exact sum plus ``O(P^2 u^2) (|start| + sum |x_p|)`` (``u = 2^-24``).  The
    deterministic package sums the same partials in FP64 and rounds once, so
    ``dfn`` is the deterministic ``dfn`` except where the exact sum lies
    within that last term of an FP32 rounding boundary (see the module
    docstring).

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams of the forward
        (``H = X.shape[2]``).
    dY : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous gradient of ``Y``.
    dU : torch.Tensor
        BF16 ``[T, H]`` contiguous gradient of ``U`` from the branch backward.
    B : torch.Tensor, optional
        BF16 ``[T, H]`` branch output of the forward; validated when given, not read.
    fn, scale, base : torch.Tensor
        The parameters of the forward (FP32 ``[24, 4H]``, ``[3]``, ``[24]``).
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward (saved, or recomputed with
        :func:`cake_mhc_nondet_train_forward_stats`).
    grad_stats : torch.Tensor
        FP32 ``[T, 32]`` record of :func:`cake_mhc_nondet_train_backward_prepare`.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    accumulate : bool
        ``False`` (default): ``dfn``, ``dscale`` and ``dbase`` are written.
        ``True``: the caller passes all three and the sums are added to their
        current values.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_nondet_train_backward_workspace`;
        a per-device default when omitted.
    dX : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous output.
    dfn, dscale, dbase : torch.Tensor, optional
        Preallocated FP32 ``[24, 4H]``, ``[3]``, ``[24]`` contiguous outputs
        (required with ``accumulate=True``).

    Returns
    -------
    ``(dX, dfn, dscale, dbase)``: the BF16 input gradient and the FP32
    parameter gradients.
    """

    return _pkg().cake_mhc_nondet_train_backward_finish(
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


@flashinfer_api
def cake_mhc_nondet_train_prepare(
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
):
    """Autograd-enabled ``forward_prepare``: ``(U, handle)``; pass ``handle`` to :func:`cake_mhc_nondet_train_finish`.

    The arguments and outputs are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_prepare`; the call
    makes the backend package's prepare node.  Its forward and its checkpoint
    replay are the deterministic package's bits; its backward runs
    ``backward_finish`` (see :func:`cake_mhc_nondet_train_backward_finish`).

    Parameters
    ----------
    X, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_nondet_train_forward_prepare` (``X`` and the
        parameters may require gradients).
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    save_stats : bool
        ``True`` (default): the per-token statistics are kept for the backward.
        ``False``: they are recomputed from ``X`` in the backward (checkpoint
        replay), bit for bit the saved record.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (``hidden=X.shape[2]``); a per-device
        default when omitted.
    backward_workspace : torch.Tensor, optional
        Caller-owned backward workspace of :func:`cake_mhc_nondet_train_backward_workspace`;
        a per-device default when omitted.

    Returns
    -------
    ``(U, handle)``: BF16 ``U`` of shape ``[T, H]`` and the handle of this
    prepare node for one :func:`cake_mhc_nondet_train_finish`.
    """

    return _pkg().cake_mhc_nondet_train_prepare(
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


@flashinfer_api
def cake_mhc_nondet_train_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    handle,
) -> torch.Tensor:
    """Autograd-enabled ``forward_finish``: ``Y`` from ``X``, the branch output ``B`` and the prepare ``handle``.

    The arguments and output are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_finish`; ``handle``
    is the handle :func:`cake_mhc_nondet_train_prepare` returned (any other
    handle, such as a ``cake_mhc_det_train_prepare`` one, raises ``TypeError``).

    Parameters
    ----------
    X : torch.Tensor
        The BF16 ``[T, 4, H]`` tensor passed to :func:`cake_mhc_nondet_train_prepare`.
    B : torch.Tensor
        BF16 ``[T, H]`` output of the external branch applied to ``U`` (may
        require gradients).
    handle : object
        The handle returned by :func:`cake_mhc_nondet_train_prepare`, for one finish.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, H]``.
    """

    return _pkg().cake_mhc_nondet_train_finish(X, B, handle)


# The backend objects this module re-exports, resolved on first access so that
# importing this module imports no backend.
_LAZY = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeMHCNondetFinishFunction",
    "CakeMHCNondetPrepareFunction",
    "cake_mhc_nondet_train_reset_workspace",
    "cake_mhc_nondet_train_route_key",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(_pkg(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822
    "CakeMHCNondetFinishFunction",  # noqa: F822
    "CakeMHCNondetPrepareFunction",  # noqa: F822
    "cake_mhc_nondet_train_backward_finish",
    "cake_mhc_nondet_train_backward_prepare",
    "cake_mhc_nondet_train_backward_workspace",
    "cake_mhc_nondet_train_backward_workspace_bytes",
    "cake_mhc_nondet_train_finish",
    "cake_mhc_nondet_train_forward_finish",
    "cake_mhc_nondet_train_forward_prepare",
    "cake_mhc_nondet_train_forward_stats",
    "cake_mhc_nondet_train_forward_workspace",
    "cake_mhc_nondet_train_forward_workspace_bytes",
    "cake_mhc_nondet_train_prepare",
    "cake_mhc_nondet_train_reset_workspace",  # noqa: F822
    "cake_mhc_nondet_train_route_key",  # noqa: F822
    "is_cake_mhc_nondet_train_supported",
]
