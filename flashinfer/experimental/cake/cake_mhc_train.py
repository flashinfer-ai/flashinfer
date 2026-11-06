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
SM100 / SM103 / SM107, and 5120 / 7168 on SM100 / SM103; another hidden size
raises ``ValueError``), non-deterministic: every entry point runs the
non-deterministic package ``flashinfer.experimental.cake.cake_mhc_train_backend``.
The function, the tensor contracts and the autograd wiring are those of the
deterministic package :mod:`flashinfer.cake_mhc_det_train` (documented there);
the workspaces are described below.

Every function of this module except :func:`cake_mhc_train_reset_workspace`
and :func:`route_key` takes a keyword-only ``deterministic: bool = True``, and
a call must pass ``deterministic=False`` explicitly.  Any other value, the
default ``True`` included (also ``None``, ``0``, ``1`` or ``"False"``), raises
``ValueError`` naming the deterministic package's entry point to call instead
(``flashinfer.cake_mhc_det_train.cake_mhc_det_train_*``, or
``is_cake_mhc_det_train_supported``).  Nothing falls back to the deterministic
package: a call without the keyword, which earlier ran the deterministic
package, now raises this error.

Numerical contract: ``backward_finish`` has no final reduction kernel: the
column-slab pass adds the ``dfn`` sum of every (column slab, token chunk) unit
into ``dfn`` with FP32 reductions and folds ``dscale`` / ``dbase`` in the same
grid.  Only the summation order of ``dfn`` varies from run to run: each element
lies within ``P`` FP32 roundings of the fixed-order sum of its starting value
(zero, or the caller's ``dfn`` under ``accumulate=True``) and its per-chunk
partials (``P <= 32`` token chunks; the chunk sizes are the host rule's per
token count and device, so the low-order bits can also differ from the
deterministic ``dfn``), and the FP32 reduction flushes subnormal addends and
results to zero.  ``dX`` is bitwise the deterministic value, and so are the
forward, the statistics replay and ``backward_prepare`` (copies of the
deterministic programs).  ``dscale`` and ``dbase`` are bitwise the
deterministic package's per-CTA reduction; in the token-count bands where the
deterministic package sums them over 8-token group rows instead (hidden 4096
on SM100 / SM103; see the backend's ``cake_jit``), their low-order bits can
differ from the deterministic result.

With ``torch.use_deterministic_algorithms(True)``,
:func:`cake_mhc_train_backward_finish` and the backward of a
:func:`cake_mhc_train_prepare` node raise PyTorch's alert for an operation
without a deterministic implementation right before they launch the ``dfn``
reduction (after their operands pass, and only for ``T > 0``): a
``RuntimeError``, or under ``warn_only=True`` a ``UserWarning``, after which
the non-deterministic kernels run.  The prepare node alerts also when the mode
was enabled between its forward and its backward.  ``forward_prepare``,
``forward_stats``, ``forward_finish``, ``backward_prepare``, the workspace
helpers and :func:`is_cake_mhc_train_supported` are bitwise the same from run
to run and raise no such alert or warning.

Workspaces: the forward workspace has one layout in both packages.  The
backward workspace is the deterministic one at the same offsets without the
per-chunk FP32 ``dfn`` partials (``P x 24 x 4H x 4`` bytes at hidden size
``H``).  The backward accepts any valid workspace of at least its size, so a
deterministic backward workspace of the same hidden size
(``flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_workspace``)
serves it too.  The records of either package (``stats``, ``grad_stats``) feed
the other's entry points (the same bits); a prepare handle feeds only the
finish of its own package (``TypeError`` otherwise).

Default workspaces (``workspace=None``): one buffer per device, hidden size and
power-of-two token-count bucket, shared by every stream and every graph replay
on that device; calls that can run concurrently on different streams pass
caller-owned workspaces.  No default buffer is created while the current stream
of the buffer's device is capturing a CUDA graph (``ValueError``): warm up with
the same token count and hidden size before the capture.  Every completed
launch leaves the ticket words of its workspace at zero, so a replay needs no
reset; :func:`cake_mhc_train_reset_workspace` zeroes the buffer it is given.

Name map (the deterministic names are unchanged and always deterministic)::

    flashinfer.cake_mhc_det_train                 this module
      cake_mhc_det_train_forward_prepare            cake_mhc_train_forward_prepare
      cake_mhc_det_train_forward_stats              cake_mhc_train_forward_stats
      cake_mhc_det_train_forward_finish             cake_mhc_train_forward_finish
      cake_mhc_det_train_backward_prepare           cake_mhc_train_backward_prepare
      cake_mhc_det_train_backward_finish            cake_mhc_train_backward_finish
      cake_mhc_det_train_prepare                    cake_mhc_train_prepare
      cake_mhc_det_train_finish                     cake_mhc_train_finish
      cake_mhc_det_train_{forward,backward}_        cake_mhc_train_{forward,backward}_
        workspace{,_bytes}                            workspace{,_bytes}
      cake_mhc_det_train_reset_workspace            cake_mhc_train_reset_workspace
      is_cake_mhc_det_train_supported               is_cake_mhc_train_supported
      CakeMHCDetTrainPrepareFunction                CakeMHCTrainPrepareFunction (non-deterministic)
      CakeMHCDetTrainFinishFunction                 CakeMHCTrainFinishFunction (non-deterministic)
      route_key                                     route_key

This module is the namespace of the non-deterministic mode.  The deterministic
package's functions and classes carry ``det`` in their names
(``flashinfer.cake_mhc_det_train``); 0.7.3's ``flashinfer.cake_mhc_train``
names are removed, with no alias.  The model's head
(``cake_mhc_det_train_head*``) has no cross-token reduction and no counterpart
here: use
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
    Y = cake_mhc_train_finish(X, branch(U), handle, deterministic=False)  # backward: atomic dfn

    ws_b = cake_mhc_train_backward_workspace(
        T, X.device, hidden=X.shape[2], deterministic=False)
    dX, dfn, dscale, dbase = cake_mhc_train_backward_finish(
        X, dY, dU, B, fn, scale, base, stats, grad_stats,
        workspace=ws_b, deterministic=False)
"""

from __future__ import annotations

from typing import Optional

import torch

from ...api_logging import flashinfer_experimental_api
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

_FEATURE = "Cake four-stream mHC training, non-deterministic (SM100/SM103/SM107)"


# ----------------------------------------------------------------------------
# Entry points (real ``def``s: public-API tooling reads the decorated
# signatures of this module; the bodies live in the non-deterministic package)


@flashinfer_experimental_api(feature=_FEATURE)
def is_cake_mhc_train_supported(
    device: torch.device,
    hidden: Optional[int] = None,
    *,
    deterministic: bool = True,
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has every exported route of the non-deterministic package.

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming
        ``is_cake_mhc_det_train_supported``.  See the module docstring.
    """

    _nd._require_nondeterministic(deterministic, "is_cake_mhc_train_supported")
    return _nd.is_cake_mhc_train_supported(device, hidden)


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_forward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = _nd_jit.HIDDEN,
    deterministic: bool = True,
) -> int:
    """Bytes of the caller-owned forward workspace for up to ``T`` tokens on ``device``.

    The forward workspace has one layout in both packages (see
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_forward_workspace_bytes`).
    ``hidden`` is the hidden size of the calls the buffer serves: 4096, 5120 or
    7168 on SM100 / SM103, 4096 on SM107 (any other value raises ``ValueError``).

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

    return _nd.cake_mhc_train_forward_workspace_bytes(
        T, device, hidden=hidden, deterministic=deterministic
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_backward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = _nd_jit.HIDDEN,
    deterministic: bool = True,
) -> int:
    """Bytes of the caller-owned backward workspace for up to ``T`` tokens on ``device``.

    The non-deterministic backward's layout: the deterministic package's
    regions at the same offsets without the per-chunk FP32 ``dfn`` partials
    (``P x 24 x 4H x 4`` bytes at hidden size ``H``, ``P <= 32``).  A
    deterministic backward workspace of the same hidden size
    (:func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_workspace`)
    also serves the non-deterministic backward.  The size is the maximum over
    every token count up to ``T``; ``T == 0`` needs no workspace.  ``hidden``
    as in :func:`cake_mhc_train_forward_workspace_bytes`.

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

    return _nd.cake_mhc_train_backward_workspace_bytes(
        T, device, hidden=hidden, deterministic=deterministic
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_forward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = _nd_jit.HIDDEN,
    deterministic: bool = True,
) -> torch.Tensor:
    """Allocate a zero-initialized forward workspace (one layout in both packages; see
    :func:`cake_mhc_train_forward_workspace_bytes`).

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

    return _nd.cake_mhc_train_forward_workspace(
        T, device, hidden=hidden, deterministic=deterministic
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_backward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = _nd_jit.HIDDEN,
    deterministic: bool = True,
) -> torch.Tensor:
    """Allocate a zero-initialized backward workspace of the non-deterministic layout (see
    :func:`cake_mhc_train_backward_workspace_bytes`).

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

    return _nd.cake_mhc_train_backward_workspace(
        T, device, hidden=hidden, deterministic=deterministic
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
    The call runs the non-deterministic package, whose forward runs copies of
    the deterministic programs, so ``U`` and ``stats`` are the deterministic
    package's bits.

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

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
        deterministic=deterministic,
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
    the call runs the non-deterministic package (the deterministic package's
    bits).

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

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
        deterministic=deterministic,
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
    the call runs the non-deterministic package (the deterministic package's
    bits).

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

    return _nd.cake_mhc_train_forward_finish(
        X, B, stats, out=out, deterministic=deterministic
    )


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
    the call runs the non-deterministic package (the deterministic package's
    bits: this pass has no cross-token reduction).

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

    return _nd.cake_mhc_train_backward_prepare(
        dY,
        B,
        X,
        stats,
        dB=dB,
        grad_stats_out=grad_stats_out,
        deterministic=deterministic,
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
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_finish`;
    the call runs the non-deterministic backward.  ``dfn`` is summed with FP32
    reductions in a run-to-run varying order, each element within ``P <= 32``
    FP32 roundings of the fixed-order sum of its starting value (zero, or the
    caller's ``dfn`` under ``accumulate=True``) and its per-chunk partials (the
    chunk sizes are the host rule's per token count and device, so the
    low-order bits can also differ from the deterministic ``dfn``); ``dX`` is
    bitwise the deterministic value; ``dscale`` and ``dbase`` are bitwise the
    deterministic package's per-CTA reduction (in its group-row bands their
    low-order bits can differ from the deterministic result; see the module
    docstring); ``accumulate=True`` adds onto the caller's values the same way.  Under
    ``torch.use_deterministic_algorithms(True)`` the call raises PyTorch's
    alert right before its launches (``RuntimeError``; under
    ``warn_only=True`` a ``UserWarning``, and the call runs).

    Parameters
    ----------
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_train_backward_workspace`,
        or a deterministic backward workspace of the same hidden size
        (:func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_workspace`);
        a per-device default when omitted.
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

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
        deterministic=deterministic,
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
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_prepare`; the call
    makes the non-deterministic package's prepare node.  Its forward and its
    checkpoint replay are the deterministic package's bits; its backward runs
    the non-deterministic ``backward_finish`` (see
    :func:`cake_mhc_train_backward_finish`) and raises PyTorch's alert when the
    deterministic algorithms are enabled at that time, also when they were
    enabled after the forward (``RuntimeError``; under ``warn_only=True`` a
    ``UserWarning``, and the backward runs).

    Parameters
    ----------
    backward_workspace : torch.Tensor, optional
        Caller-owned backward workspace of :func:`cake_mhc_train_backward_workspace`,
        or a deterministic backward workspace of the same hidden size; a
        per-device default when omitted.
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

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
        deterministic=deterministic,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_mhc_train_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    handle,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """Autograd-enabled ``forward_finish``: ``Y`` from ``X``, the branch output ``B`` and the prepare ``handle``.

    The arguments and output are those of
    :func:`~flashinfer.cake_mhc_det_train.cake_mhc_det_train_finish`; ``handle``
    is the handle :func:`cake_mhc_train_prepare` returned (any other handle,
    such as a ``cake_mhc_det_train_prepare`` one, raises ``TypeError``).

    Parameters
    ----------
    deterministic : bool
        Must be ``False`` (pass it explicitly); any other value, the default
        included, raises ``ValueError`` naming the ``cake_mhc_det_train_*``
        entry.  See the module docstring.
    """

    return _nd.cake_mhc_train_finish(X, B, handle, deterministic=deterministic)


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
