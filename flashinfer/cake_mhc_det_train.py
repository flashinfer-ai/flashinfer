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

Cake four-stream mHC training forward / backward (4 streams; hidden size
``H`` in {4096, 5120, 7168}, taken per call from ``X.shape[2]``; SM100 / SM103
/ SM107 at all three sizes).

The function (FP32 arithmetic, BF16 activation boundaries)::

    u = float32(X).reshape(T, 4 * H)
    r = rsqrt(mean(u * u, axis=-1) + 1e-5)
    z = (u * r) @ fn.T                                   # [T, 24]
    pre  = sigmoid(z[:, :4]  * scale[0] + base[:4]) + 1e-6
    post = 2 * sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    C = row_softmax((z[:, 8:] * scale[2] + base[8:]).reshape(T, 4, 4)) + 1e-6
    C = C / (column_sum(C) + 1e-6); 19 x { row-normalize; column-normalize }
    U[t, d] = BF16(sum_i pre[t, i] X[t, i, d])
    B = external_branch(U)                               # attention / MLP, BF16 [T, H]
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
``cake_mhc_det_train_prepare(..., save_stats=False)`` the statistics are instead
recomputed from ``X`` in the backward with bitwise the same gradients
(checkpoint replay).  The finish node's backward hands ``dY`` and the gradient
statistics to the prepare node's backward on autograd edges, so nothing of a
connection stays alive once its backward has run; at hidden 4096, up to 4096
tokens on SM100 / SM107 and up to 2048 tokens on SM103, the replay also writes
the transposed split of ``fn`` (3 MiB), handed on one more edge, so the
backward does not split ``fn`` again.  Workspaces are caller-held
(``*_workspace_bytes`` / ``*_workspace``) and, with preallocated outputs,
the submit path allocates nothing; a workspace serves the calls of the
hidden size it was sized for
(``hidden=``, default 4096) up to ``T`` tokens.  All tensors are contiguous:
``X``, ``Y``, ``dY``, ``dX`` BF16 ``[T, 4, H]``; ``U``, ``B``, ``dU``, ``dB``
BF16 ``[T, H]``; ``fn`` FP32 ``[24, 4H]``; ``scale`` FP32 ``[3]``; ``base``
FP32 ``[24]``.  The token count ``T`` is a runtime scalar: it may change
between calls, ``T == 1`` is supported and ``T == 0`` returns empty outputs
without launching.  Calls of different hidden sizes may be mixed in one
process.  Every behaviour
switch is a keyword argument with its default in the docstring.

Names and mode: this is the deterministic module.  Its backend package
``flashinfer.experimental.cake.cake_mhc_det_train`` is imported on the first
call or attribute access (importing this module imports no backend).  Every
name this module exports carries the mode word ``det``
(``cake_mhc_det_train_*``, ``CakeMHCDet*Function``) except the constants
``ARCHES`` and ``SUPPORTED_COMPUTE_CAPABILITIES``; former names have no alias
(FlashInfer 0.7.3's ``flashinfer.cake_mhc_train`` names,
``CakeMHCDetTrain{Prepare,Finish}Function`` and ``route_key`` fail at import).
The entry points take no ``deterministic`` keyword, give the same results
under ``torch.use_deterministic_algorithms(True)`` and raise no alert.  The
non-deterministic mode is ``flashinfer.cake_mhc_nondet_train``.

Usage::

    U, stats = cake_mhc_det_train_forward_prepare(X, fn, scale, base)
    B = branch(U)
    Y = cake_mhc_det_train_forward_finish(X, B, stats)
    dB, grad_stats = cake_mhc_det_train_backward_prepare(dY, B, X, stats)
    dU = branch_backward(dB)
    dX, dfn, dscale, dbase = cake_mhc_det_train_backward_finish(
        X, dY, dU, B, fn, scale, base, stats, grad_stats)

    # autograd around the external branch
    U, handle = cake_mhc_det_train_prepare(X, fn, scale, base)      # save_stats=False: checkpoint replay
    Y = cake_mhc_det_train_finish(X, branch(U), handle)

    # caller-owned workspaces (zero-initialized; the kernels leave them reusable)
    ws_f = cake_mhc_det_train_forward_workspace(T, X.device, hidden=X.shape[2])
    ws_b = cake_mhc_det_train_backward_workspace(T, X.device, hidden=X.shape[2])

The model's head, the stream mean ``Y = mean_i X[:, i, :]`` of the last
layer's four streams, is the module :mod:`flashinfer.cake_mhc_head_det_train`.
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
    from .experimental.cake import cake_mhc_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_mhc_det_train


# The stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources, imported on the first call.


@flashinfer_api
def cake_mhc_det_train_forward_workspace_bytes(
    T: int, device: torch.device, *, hidden: int = _HIDDEN
) -> int:
    r"""Bytes of the caller-owned forward workspace for up to ``T`` tokens of hidden size ``hidden``.

    The forward routes (``forward_prepare`` and the ``forward_stats`` replay)
    hold one 256-byte ticket region (first, at the same offset for every token
    count), the BF16 pieces of ``fn`` in the mapping operand layout and the
    FP32 per-K-unit partials of the mapping pass.  The ticket words must be
    zero on first use (allocate with ``torch.zeros`` or
    :func:`cake_mhc_det_train_forward_workspace`); the kernels leave them zero.

    Parameters
    ----------
    T : int
        Largest token count the buffer serves.  The size is the maximum over
        every token count up to ``T``, so the buffer also serves every smaller
        one (and CUDA graph replays at a smaller ``T``); ``T == 0`` needs no
        workspace (nothing is launched).
    device : torch.device
        CUDA device of the calls.
    hidden : int
        Hidden size ``H`` of the calls the buffer serves (``X.shape[2]``;
        default ``4096``): a hidden size the device's architecture exports
        (4096, 5120 or 7168 on SM100 / SM103 / SM107), else
        ``ValueError``.  The ``fn`` pieces and the partials scale with ``4 H``;
        every call checks the buffer against its own ``H`` and ``T``, so a
        buffer sized for one ``H`` is refused (``ValueError``) by a call of a
        larger ``H`` it is too small for.

    Returns
    -------
    The size of the forward workspace in bytes.
    """

    return _pkg().cake_mhc_det_train_forward_workspace_bytes(T, device, hidden=hidden)


@flashinfer_api
def cake_mhc_det_train_backward_workspace_bytes(
    T: int, device: torch.device, *, hidden: int = _HIDDEN
) -> int:
    r"""Bytes of the caller-owned backward workspace for up to ``T`` tokens of hidden size ``hidden``.

    The ``backward_finish`` route holds one 256-byte ticket region (first, at
    the same offset for every token count), the transposed BF16 pieces of
    ``fn``, the per-token records of the token-owner pass (``dz'`` pieces and
    the per-token scalars, padded to a multiple of 128 rows), the per-CTA
    parameter partials and the per-chunk FP32 partials of ``dfn``.  The ticket
    words must be zero on first use (allocate with ``torch.zeros`` or
    :func:`cake_mhc_det_train_backward_workspace`); the kernels leave them zero.

    Parameters
    ----------
    T : int
        Largest token count the buffer serves (the size is the maximum over
        every token count up to ``T``); ``T == 0`` needs no workspace.
    device : torch.device
        CUDA device of the calls.
    hidden : int
        Hidden size ``H`` of the calls the buffer serves (``X.shape[2]``;
        default ``4096``), as in
        :func:`cake_mhc_det_train_forward_workspace_bytes`: a buffer sized for
        one ``H`` is refused by a call of a larger ``H`` it is too small for.

    Returns
    -------
    The size of the backward workspace in bytes.
    """

    return _pkg().cake_mhc_det_train_backward_workspace_bytes(T, device, hidden=hidden)


@flashinfer_api
def cake_mhc_det_train_forward_prepare(
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
        BF16 ``[T, 4, H]`` contiguous residual streams (``T`` is a runtime
        scalar; ``T == 0`` returns empty outputs without launching).  The
        hidden size ``H = X.shape[2]`` is 4096, 5120 or 7168 on SM100 / SM103
        / SM107.
    fn : torch.Tensor
        FP32 ``[24, 4H]`` contiguous mapping matrix.
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
        Preallocated BF16 ``[T, H]`` contiguous ``U``.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_det_train_forward_workspace`
        (``hidden=X.shape[2]``); a per-device default buffer is used when omitted.

    Returns
    -------
    ``(U, stats)``: BF16 ``U[t, d] = BF16(sum_i pre[t, i] X[t, i, d])`` and the
    FP32 ``[T, 64]`` per-token record ``r | z[24] | pre[4] | post[4] | C[16] | pad[15]``
    (256 bytes per token) that the finish and backward entry points consume.
    """

    return _pkg().cake_mhc_det_train_forward_prepare(
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
def cake_mhc_det_train_forward_stats(
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
        BF16 ``[T, 4, H]`` contiguous residual streams.
    fn : torch.Tensor
        FP32 ``[24, 4H]`` contiguous mapping matrix.
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
        :func:`cake_mhc_det_train_forward_prepare`).
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (:func:`cake_mhc_det_train_forward_workspace`,
        ``hidden=X.shape[2]``).

    Returns
    -------
    FP32 ``[T, 64]`` ``stats``, bitwise identical to the record
    :func:`cake_mhc_det_train_forward_prepare` returns for the same inputs on the
    same device (the same work partition without the ``U`` write).
    """

    return _pkg().cake_mhc_det_train_forward_stats(
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
def cake_mhc_det_train_forward_finish(
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
        BF16 ``[T, 4, H]`` contiguous residual streams (the ``forward_prepare`` input).
    B : torch.Tensor
        BF16 ``[T, H]`` contiguous output of the external branch applied to ``U``.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record returned by :func:`cake_mhc_det_train_forward_prepare`
        (or :func:`cake_mhc_det_train_forward_stats`).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous ``Y``.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, H]``; the FP32 sum is rounded once.
    """

    return _pkg().cake_mhc_det_train_forward_finish(X, B, stats, out=out)


@flashinfer_api
def cake_mhc_det_train_backward_prepare(
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
        BF16 ``[T, 4, H]`` contiguous gradient of ``Y``.
    B : torch.Tensor
        BF16 ``[T, H]`` contiguous branch output of the forward.
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams of the forward
        (``H = X.shape[2]``).
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward.
    dB : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous output.
    grad_stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 32]`` contiguous output.

    Returns
    -------
    tuple of torch.Tensor
        ``(dB, grad_stats)``: ``dB[t, d] = BF16(sum_j post[t, j] dY[t, j, d])`` (one
        rounding of the full FP32 sum) and the FP32 ``[T, 32]`` record
        ``dpost[4] | dC[16] | zeros[12]`` that :func:`cake_mhc_det_train_backward_finish` consumes.
    """

    return _pkg().cake_mhc_det_train_backward_prepare(
        dY, B, X, stats, dB=dB, grad_stats_out=grad_stats_out
    )


@flashinfer_api
def cake_mhc_det_train_backward_finish(
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
        BF16 ``[T, 4, H]`` contiguous residual streams of the forward
        (``H = X.shape[2]``).
    dY : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous gradient of ``Y``.
    dU : torch.Tensor
        BF16 ``[T, H]`` contiguous gradient of ``U`` from the branch backward.
    B : torch.Tensor, optional
        BF16 ``[T, H]`` branch output of the forward; accepted for the
        prepare / finish symmetry and validated when given, not read.
    fn : torch.Tensor
        FP32 ``[24, 4H]`` contiguous mapping matrix of the forward.
    scale : torch.Tensor
        FP32 ``[3]`` group scales of the forward.
    base : torch.Tensor
        FP32 ``[24]`` logit offsets of the forward.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward (saved, or recomputed with
        :func:`cake_mhc_det_train_forward_stats`; both give bitwise the same gradients).
    grad_stats : torch.Tensor
        FP32 ``[T, 32]`` record returned by :func:`cake_mhc_det_train_backward_prepare`.
    rms_eps : float
        Normalization epsilon (default ``1e-5``; a launch argument).
    hc_eps : float
        Coefficient epsilon (default ``1e-6``; a launch argument).
    sinkhorn_iters : int
        Sinkhorn iteration count, 1 to 64 (default ``20``; as in
        :func:`cake_mhc_det_train_forward_prepare`).
    accumulate : bool
        ``False`` (default): ``dfn``, ``dscale`` and ``dbase`` are written.
        ``True``: the caller passes all three and the fixed-order sums are added
        to their current values (reproducible across consecutive accumulations).
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_det_train_backward_workspace`
        (``hidden=X.shape[2]``); a per-device default buffer is used when omitted.
    dX : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous output.
    dfn : torch.Tensor, optional
        Preallocated FP32 ``[24, 4H]`` contiguous output (required with ``accumulate=True``).
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

    return _pkg().cake_mhc_det_train_backward_finish(
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
def cake_mhc_det_train_prepare(
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
    """Autograd-enabled ``forward_prepare``: ``(U, handle)``; pass ``handle`` to :func:`cake_mhc_det_train_finish`.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams (may require gradients;
        ``H = X.shape[2]``).
    fn : torch.Tensor
        FP32 ``[24, 4H]`` contiguous mapping matrix (may require gradients).
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
        :func:`cake_mhc_det_train_forward_prepare`).
    save_stats : bool
        ``True`` (default): the per-token statistics are kept for the backward.
        ``False``: they are dropped after the forward and recomputed from ``X``
        in the backward (checkpoint replay); the gradients are bitwise the same.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (``hidden=X.shape[2]``); a per-device
        default when omitted.
    backward_workspace : torch.Tensor, optional
        Caller-owned backward workspace (``hidden=X.shape[2]``); a per-device
        default when omitted.

    Returns
    -------
    ``(U, handle)``: BF16 ``U`` of shape ``[T, H]`` and the handle of this
    prepare node for one :func:`cake_mhc_det_train_finish`.  The backward of the
    prepare node returns the single-cast ``dX`` and FP32 ``dfn``, ``dscale``,
    ``dbase``.
    """

    return _pkg().cake_mhc_det_train_prepare(
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
def cake_mhc_det_train_finish(X: torch.Tensor, B: torch.Tensor, handle) -> torch.Tensor:
    """Autograd-enabled ``forward_finish``: ``Y`` from ``X``, the branch output ``B`` and the prepare ``handle``.

    Parameters
    ----------
    X : torch.Tensor
        The BF16 ``[T, 4, H]`` tensor passed to :func:`cake_mhc_det_train_prepare`.
    B : torch.Tensor
        BF16 ``[T, H]`` output of the external branch applied to ``U`` (may require gradients).
    handle : object
        The handle returned by :func:`cake_mhc_det_train_prepare`, for one finish.  A
        reentrant checkpoint region (``use_reentrant=True``) holds both calls or
        starts after the prepare; it does not end between them.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, H]``.  Its backward returns ``dB`` for the
    branch and no gradient for ``X`` (the residual-mixing contribution is part
    of the single-cast ``dX`` of the prepare node).
    """

    return _pkg().cake_mhc_det_train_finish(X, B, handle)


# The backend objects this module re-exports, resolved on first access so that
# importing this module imports no backend.
_LAZY = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeMHCDetFinishFunction",
    "CakeMHCDetPrepareFunction",
    "cake_mhc_det_train_backward_workspace",
    "cake_mhc_det_train_forward_workspace",
    "cake_mhc_det_train_reset_workspace",
    "cake_mhc_det_train_route_key",
    "is_cake_mhc_det_train_supported",
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
    "CakeMHCDetFinishFunction",  # noqa: F822
    "CakeMHCDetPrepareFunction",  # noqa: F822
    "cake_mhc_det_train_backward_finish",
    "cake_mhc_det_train_backward_prepare",
    "cake_mhc_det_train_backward_workspace",  # noqa: F822
    "cake_mhc_det_train_backward_workspace_bytes",
    "cake_mhc_det_train_finish",
    "cake_mhc_det_train_forward_finish",
    "cake_mhc_det_train_forward_prepare",
    "cake_mhc_det_train_forward_stats",
    "cake_mhc_det_train_forward_workspace",  # noqa: F822
    "cake_mhc_det_train_forward_workspace_bytes",
    "cake_mhc_det_train_prepare",
    "cake_mhc_det_train_reset_workspace",  # noqa: F822
    "cake_mhc_det_train_route_key",  # noqa: F822
    "is_cake_mhc_det_train_supported",  # noqa: F822
]
