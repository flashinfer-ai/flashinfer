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

Cake four-stream mHC training forward / backward (hidden 4096, 4 streams;
SM100 / SM103 / SM107).

The function (FP32 arithmetic, BF16 activation boundaries)::

    u = float32(X).reshape(T, 16384)
    r = rsqrt(mean(u * u, axis=-1) + 1e-5)
    z = (u * r) @ fn.T                                   # [T, 24]
    pre  = sigmoid(z[:, :4]  * scale[0] + base[:4]) + 1e-6
    post = 2 * sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    C = row_softmax((z[:, 8:] * scale[2] + base[8:]).reshape(T, 4, 4)) + 1e-6
    C = C / (column_sum(C) + 1e-6); 19 x { row-normalize; column-normalize }
    U[t, d] = BF16(sum_i pre[t, i] X[t, i, d])
    B = external_branch(U)                               # attention / MLP, BF16 [T, 4096]
    Y[t, j, d] = BF16(post[t, j] B[t, d] + sum_i C[t, i, j] X[t, i, d])

Forward: ``forward_prepare`` (``U`` and a 256-byte FP32 statistics record per
token: ``r``, ``z``, ``pre``, ``post``, ``C``), the external branch,
``forward_finish`` (``Y``).  Backward: ``backward_prepare`` before the branch
backward (``dB = BF16(sum_j post_j dY_j)`` with one rounding of the full FP32
sum, plus per-token gradient statistics), the branch backward (``dU``),
``backward_finish`` (``dX`` with the mapping, collapse and residual-mixing
contributions summed in FP32 and rounded once; FP32 ``dfn``, ``dscale``,
``dbase`` by fixed-order reductions, bitwise reproducible for a given token
count on a given device, optionally accumulated into caller buffers).  The
mapping arithmetic is FP32-quality throughout (exact three-way BF16 split of
``fn``, FP32 accumulation); the sigmoid tails keep their gradients where the
FP32 forward sigmoid rounds to one.

Retained between forward and backward: the inputs and ``stats`` (256 bytes
per token), all through the autograd saved tensors, so an activation
checkpoint (``torch.utils.checkpoint``) frees them after the forward and
recomputes them in the backward; with
``cake_mhc_train_prepare(..., save_stats=False)`` the statistics are instead
recomputed from ``X`` in the backward with bitwise the same gradients
(checkpoint replay).  The finish node's backward hands ``dY`` and the gradient
statistics to the prepare node's backward on autograd edges, so nothing of a
connection stays alive once its backward has run.  Workspaces are caller-held
(``*_workspace_bytes`` / ``*_workspace``) and the submit path allocates
nothing.  All tensors are contiguous: ``X``, ``Y``, ``dY``, ``dX`` BF16
``[T, 4, 4096]``; ``U``, ``B``, ``dU``, ``dB`` BF16 ``[T, 4096]``; ``fn`` FP32
``[24, 16384]``; ``scale`` FP32 ``[3]``; ``base`` FP32 ``[24]``.  The token
count ``T`` is a runtime scalar: it may change between calls, ``T == 1`` is
supported and ``T == 0`` returns empty outputs without launching.  Every
behaviour switch is a keyword argument with its default in the docstring.

Usage::

    U, stats = cake_mhc_train_forward_prepare(X, fn, scale, base)
    B = branch(U)
    Y = cake_mhc_train_forward_finish(X, B, stats)
    dB, grad_stats = cake_mhc_train_backward_prepare(dY, B, X, stats)
    dU = branch_backward(dB)
    dX, dfn, dscale, dbase = cake_mhc_train_backward_finish(
        X, dY, dU, B, fn, scale, base, stats, grad_stats)

    # autograd around the external branch
    U, handle = cake_mhc_train_prepare(X, fn, scale, base)      # save_stats=False: checkpoint replay
    Y = cake_mhc_train_finish(X, branch(U), handle)

    # caller-owned workspaces (zero-initialized; the kernels leave them reusable)
    ws_f = cake_mhc_train_forward_workspace(T, X.device)
    ws_b = cake_mhc_train_backward_workspace(T, X.device)
"""

from __future__ import annotations

from typing import Optional

import torch

from . import cake_mhc_train_backend as _backend
from .api_logging import flashinfer_api
from .cake_mhc_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeMHCTrainFinishFunction,
    CakeMHCTrainPrepareFunction,
    cake_mhc_train_backward_workspace,
    cake_mhc_train_backward_workspace_bytes,
    cake_mhc_train_forward_workspace,
    cake_mhc_train_forward_workspace_bytes,
    cake_mhc_train_reset_workspace,
    is_cake_mhc_train_supported,
    route_key,
)

# The stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
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
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Forward mapping, coefficients and collapse: ``U`` and the per-token statistics.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams (``T`` is a runtime
        scalar; ``T == 0`` returns empty outputs without launching).
    fn : torch.Tensor
        FP32 ``[24, 16384]`` contiguous mapping matrix.
    scale : torch.Tensor
        FP32 ``[3]`` group scales of the pre / post / Sinkhorn logits.
    base : torch.Tensor
        FP32 ``[24]`` logit offsets.
    rms_eps : float
        Normalization epsilon (default ``1e-5``): a launch argument of the exported
        kernels, so any finite positive value runs on the same compiled programs.
    hc_eps : float
        Coefficient epsilon (default ``1e-6``): a launch argument like ``rms_eps``.
    sinkhorn_iters : int
        Sinkhorn iteration count, 1 to 64 (default ``20``).  The default runs the
        compiled programs, whose chain is unrolled for it; every other count runs
        the run-time-count programs of the coefficient and token-owner passes.  A
        count above 64 raises ``NotImplementedError``.
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, 4096]`` contiguous ``U``.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_train_forward_workspace`;
        a per-device default buffer is used when omitted.

    Returns
    -------
    ``(U, stats)``: BF16 ``U[t, d] = BF16(sum_i pre[t, i] X[t, i, d])`` and the
    FP32 ``[T, 64]`` per-token record ``r | z[24] | pre[4] | post[4] | C[16] | pad[15]``
    (256 bytes per token) that the finish and backward entry points consume.
    """

    return _backend.cake_mhc_train_forward_prepare(
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
) -> torch.Tensor:
    r"""Recompute the per-token statistics of ``forward_prepare`` (checkpoint replay).

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams.
    fn : torch.Tensor
        FP32 ``[24, 16384]`` contiguous mapping matrix.
    scale : torch.Tensor
        FP32 ``[3]`` group scales.
    base : torch.Tensor
        FP32 ``[24]`` logit offsets.
    rms_eps : float
        Normalization epsilon (default ``1e-5``; a launch argument).
    hc_eps : float
        Coefficient epsilon (default ``1e-6``; a launch argument).
    sinkhorn_iters : int
        Sinkhorn iteration count, 1 to 64 (default ``20``; as in
        :func:`cake_mhc_train_forward_prepare`).
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (:func:`cake_mhc_train_forward_workspace`).

    Returns
    -------
    FP32 ``[T, 64]`` ``stats``, bitwise identical to the record
    :func:`cake_mhc_train_forward_prepare` returns for the same inputs on the
    same device (the same work partition without the ``U`` write).
    """

    return _backend.cake_mhc_train_forward_stats(
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
def cake_mhc_train_forward_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    stats: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Residual mixing: ``Y[t, j, d] = BF16(post[t, j] B[t, d] + sum_i C[t, i, j] X[t, i, d])``.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams (the ``forward_prepare`` input).
    B : torch.Tensor
        BF16 ``[T, 4096]`` contiguous output of the external branch applied to ``U``.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record returned by :func:`cake_mhc_train_forward_prepare`
        (or :func:`cake_mhc_train_forward_stats`).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, 4096]`` contiguous ``Y``.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, 4096]``; the FP32 sum is rounded once.
    """

    return _backend.cake_mhc_train_forward_finish(X, B, stats, out=out)


@flashinfer_api
def cake_mhc_train_backward_prepare(
    dY: torch.Tensor,
    B: torch.Tensor,
    X: torch.Tensor,
    stats: torch.Tensor,
    *,
    dB: Optional[torch.Tensor] = None,
    grad_stats_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Branch-output gradient and the per-token gradient statistics (before the branch backward).

    Parameters
    ----------
    dY : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous gradient of ``Y``.
    B : torch.Tensor
        BF16 ``[T, 4096]`` contiguous branch output of the forward.
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams of the forward.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward.
    dB : torch.Tensor, optional
        Preallocated BF16 ``[T, 4096]`` contiguous output.
    grad_stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 32]`` contiguous output.

    Returns
    -------
    ``(dB, grad_stats)``: ``dB[t, d] = BF16(sum_j post[t, j] dY[t, j, d])`` (one
    rounding of the full FP32 sum) and the FP32 ``[T, 32]`` record
    ``dpost[4] | dC[16] | zeros[12]`` that :func:`cake_mhc_train_backward_finish` consumes.
    """

    return _backend.cake_mhc_train_backward_prepare(
        dY, B, X, stats, dB=dB, grad_stats_out=grad_stats_out
    )


@flashinfer_api
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Input gradient and parameter gradients (after the branch backward produced ``dU``).

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams of the forward.
    dY : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous gradient of ``Y``.
    dU : torch.Tensor
        BF16 ``[T, 4096]`` contiguous gradient of ``U`` from the branch backward.
    B : torch.Tensor, optional
        BF16 ``[T, 4096]`` branch output of the forward; accepted for the
        prepare / finish symmetry and validated when given, not read.
    fn : torch.Tensor
        FP32 ``[24, 16384]`` contiguous mapping matrix of the forward.
    scale : torch.Tensor
        FP32 ``[3]`` group scales of the forward.
    base : torch.Tensor
        FP32 ``[24]`` logit offsets of the forward.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward (saved, or recomputed with
        :func:`cake_mhc_train_forward_stats`; both give bitwise the same gradients).
    grad_stats : torch.Tensor
        FP32 ``[T, 32]`` record returned by :func:`cake_mhc_train_backward_prepare`.
    rms_eps : float
        Normalization epsilon (default ``1e-5``; a launch argument).
    hc_eps : float
        Coefficient epsilon (default ``1e-6``; a launch argument).
    sinkhorn_iters : int
        Sinkhorn iteration count, 1 to 64 (default ``20``; as in
        :func:`cake_mhc_train_forward_prepare`).
    accumulate : bool
        ``False`` (default): ``dfn``, ``dscale`` and ``dbase`` are written.
        ``True``: the caller passes all three and the fixed-order sums are added
        to their current values (reproducible across consecutive accumulations).
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_train_backward_workspace`;
        a per-device default buffer is used when omitted.
    dX : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, 4096]`` contiguous output.
    dfn : torch.Tensor, optional
        Preallocated FP32 ``[24, 16384]`` contiguous output (required with ``accumulate=True``).
    dscale : torch.Tensor, optional
        Preallocated FP32 ``[3]`` output (required with ``accumulate=True``).
    dbase : torch.Tensor, optional
        Preallocated FP32 ``[24]`` output (required with ``accumulate=True``).

    Returns
    -------
    ``(dX, dfn, dscale, dbase)``: the BF16 ``dX`` sums the mapping, collapse and
    residual-mixing contributions in FP32 and rounds once; the FP32 parameter
    gradients are fixed-order reductions.  With ``T == 0`` nothing is launched:
    ``dX`` is empty and the parameter gradients are zeroed (``accumulate=False``)
    or left untouched (``accumulate=True``).
    """

    return _backend.cake_mhc_train_backward_finish(
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
):
    """Autograd-enabled ``forward_prepare``: ``(U, handle)``; pass ``handle`` to :func:`cake_mhc_train_finish`.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams (may require gradients).
    fn : torch.Tensor
        FP32 ``[24, 16384]`` contiguous mapping matrix (may require gradients).
    scale : torch.Tensor
        FP32 ``[3]`` group scales (may require gradients).
    base : torch.Tensor
        FP32 ``[24]`` logit offsets (may require gradients).
    rms_eps : float
        Normalization epsilon (default ``1e-5``; a launch argument).
    hc_eps : float
        Coefficient epsilon (default ``1e-6``; a launch argument).
    sinkhorn_iters : int
        Sinkhorn iteration count, 1 to 64 (default ``20``; as in
        :func:`cake_mhc_train_forward_prepare`).
    save_stats : bool
        ``True`` (default): the per-token statistics are kept for the backward.
        ``False``: they are dropped after the forward and recomputed from ``X``
        in the backward (checkpoint replay); the gradients are bitwise the same.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace; a per-device default when omitted.
    backward_workspace : torch.Tensor, optional
        Caller-owned backward workspace; a per-device default when omitted.

    Returns
    -------
    ``(U, handle)``: BF16 ``U`` of shape ``[T, 4096]`` and the handle of this
    prepare node for one :func:`cake_mhc_train_finish`.  The backward of the
    prepare node returns the single-cast ``dX`` and FP32 ``dfn``, ``dscale``,
    ``dbase``.
    """

    return _backend.cake_mhc_train_prepare(
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
def cake_mhc_train_finish(X: torch.Tensor, B: torch.Tensor, handle) -> torch.Tensor:
    """Autograd-enabled ``forward_finish``: ``Y`` from ``X``, the branch output ``B`` and the prepare ``handle``.

    Parameters
    ----------
    X : torch.Tensor
        The BF16 ``[T, 4, 4096]`` tensor passed to :func:`cake_mhc_train_prepare`.
    B : torch.Tensor
        BF16 ``[T, 4096]`` output of the external branch applied to ``U`` (may require gradients).
    handle : object
        The handle returned by :func:`cake_mhc_train_prepare`, for one finish.  A
        reentrant checkpoint region (``use_reentrant=True``) holds both calls or
        starts after the prepare; it does not end between them.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, 4096]``.  Its backward returns ``dB`` for the
    branch and no gradient for ``X`` (the residual-mixing contribution is part
    of the single-cast ``dX`` of the prepare node).
    """

    return _backend.cake_mhc_train_finish(X, B, handle)


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
