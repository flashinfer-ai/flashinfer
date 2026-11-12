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

Cake fused BF16 RMSNorm training forward / backward (SM100 / SM103 / SM107).

Mode and dispatch rules:

* The mode is part of the name, and only of the name.  This is the
  deterministic package: its weight gradient ``dw`` is bitwise reproducible
  and every public name carries ``det`` (the constants ``ARCHES`` and
  ``SUPPORTED_COMPUTE_CAPABILITIES`` excepted).  No entry point takes a
  ``deterministic`` keyword; passing one raises Python's ``TypeError`` for an
  unexpected keyword argument.
* The non-deterministic package is :mod:`flashinfer.cake_rmsnorm_nondet_train`
  (names with ``nondet``; its entry points take no ``deterministic`` keyword
  either).  Neither package runs the other.
* ``torch.use_deterministic_algorithms`` changes nothing here (every launch
  is fixed-order).  The non-deterministic package's backward alerts
  under it (``RuntimeError``; ``warn_only=True``: one ``UserWarning``, then its
  own backward); its forward and helpers never alert.
* There are no aliases: a name without ``det`` never runs this package, and
  the former helper names ``route_key``, ``program_windows`` and
  ``weight_dtypes`` do not exist (use
  :func:`cake_rmsnorm_det_train_route_key`,
  :func:`cake_rmsnorm_det_train_program_windows` and
  :func:`cake_rmsnorm_det_train_weight_dtypes`).

Forward: ``r_t = (mean_j x_tj^2 + eps)^(-1/2)`` in FP32, ``y = BF16((x * r) * w)``.
The only tensor saved for the backward besides the inputs is the FP32 per-row
reciprocal RMS ``rstd`` (``[T]``).  Backward: one fused pass produces
``dx = BF16(r * g * w - x * r^3 * mean(g * w * x))`` and the FP32 weight
gradient ``dw = sum_t g * x * r`` through a fixed row-chunk partition and a
fixed-order reduction, so ``dw`` is bitwise reproducible for a given token
count on devices with the same SM count (the partition spreads the rows over
``ctas_per_sm`` x SM count CTAs; devices of one architecture with different SM
counts can give different ``dw`` bits).  The optional residual-add fusion
computes ``h_new = BF16(h + u)`` first, normalizes ``h_new``, and in the
backward adds the gradient flowing directly through ``h_new`` so that one
``dx`` tensor is the gradient of both ``h`` and ``u``.

All kernels use FP32 statistics and accumulation with IEEE division and
square root (no fast-math).  Inputs and outputs are BF16; ``rstd`` and ``dw``
are FP32.  ``w`` is BF16, or FP32 on the routes that ship the FP32-weight
programs (the effective weight ``1 + w`` of a zero-centred model, computed by
the caller in FP32; :func:`cake_rmsnorm_det_train_weight_dtypes` lists them
per route).  Row-strided inputs (``x``, ``residual``, ``g``, ``g_residual``)
are accepted when ``stride(1) == 1``, ``stride(0)`` is a multiple of eight
elements and the storage is aligned (32 bytes for the forward inputs, 16 for
the backward); outputs are contiguous.  An input of more dimensions,
``[..., H]``, is normalized row by row when its leading dimensions are
contiguous with one another in row-major order (a ``[T, heads, H]`` slice of
a wider buffer qualifies): the outputs take its shape, ``rstd`` its leading
shape, ``w`` is shared by every row and ``dw`` sums them all.  The
token count ``T`` is a runtime scalar: it may change between calls, ``T == 1``
is supported and ``T == 0`` returns empty outputs without launching.  Each
route selects one of several programs from ``T``; within one program window
a row's results do not depend on ``T`` (see "Program selection by token
count" in the backend's forward and backward notes and
:func:`cake_rmsnorm_det_train_program_windows`).

Beside kernels resident on other streams: a call's kernels (the forward
kernel; the backward main kernel, then its dw reduce kernel) start on an SM
only when one of their CTAs fits there next to the CTAs already resident.  A
kernel on another stream that keeps a CTA on every SM therefore delays the
call until it exits unless one CTA of each kernel fits next to it:

- shared memory: the SM's 228 KB of shared memory hold every resident CTA's
  allocation plus 1 KB reserved per CTA, within the carveout the SM runs at.
  An SM changes its carveout only when no CTA is resident on it;
- registers: each SM sub-partition (a quarter of the SM) holds 16 K registers
  for the warps placed on it, a warp taking 32 times its per-thread register
  count rounded up to a multiple of 8;
- threads (2048 per SM on SM100 / SM103, 1024 on SM107) and CTA slots (32 per
  SM on SM100 / SM103, 16 on SM107);
- on SM107, named barriers: a kernel that selects its named barrier at run
  time needs an SM with no other CTA.

A resident kernel with little or no shared memory and the default carveout can
hold its SMs at a small carveout; then every backward waits, and so do some
forward programs.  Setting ``cudaFuncAttributePreferredSharedMemoryCarveout``
to ``cudaSharedmemCarveoutMaxShared`` on the resident kernel removes this
wait.  The resident kernel's own shared memory still counts, so a large
dynamic shared-memory allocation makes more calls wait, not fewer: with the
max-shared carveout and one resident CTA per SM, every call outside the list
below starts while that CTA holds at most 27 KB of shared memory on SM100 /
SM103 and at most 24 KB on SM107.  Even beside one resident CTA of 128
threads without shared memory, these calls wait, with BF16 and FP32 weights
alike, because one of their kernels needs a whole SM:

- SM100 / SM103: none.
- SM107: the backward of H 1536 above 1024 tokens, the backward of H 2048
  above 256 tokens, the backward of H 4096 above 1024 tokens and the forward
  of H 6144 above 4096 tokens.

Beside those, leave some SMs free of the resident kernel.  Kernels that hold
only some SMs (communication kernels with 16-32 CTAs) do not delay the start
of any call: the call runs on the other SMs and slows down by about the share
of SMs those kernels hold.  The limits and lists above are measured on B200
(SM100), GB300 (SM103) and R200 (SM107).

Usage::

    y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)          # dw is FP32 [H]

    # autograd (dw is returned in w.dtype; the kernel accumulates in FP32)
    y = cake_rmsnorm_det(x, w, eps)
    y, h_new = cake_rmsnorm_det(h, w, eps, residual=u)                 # fused add

    # caller-owned backward workspace (partials + self-resetting counters)
    ws = cake_rmsnorm_det_train_backward_workspace(T, H, x.device)
    dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd, workspace=ws)
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api


# The backend package is imported on first use and cached: importing this module does not
# import ``flashinfer.experimental``, and a call does not repeat the import.
@functools.cache
def _backend():
    from .experimental.cake import cake_rmsnorm_det_train as backend  # noqa: PLC0415 - the backend package, imported on first use

    return backend


# The three stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
def cake_rmsnorm_det_train_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    residual: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    r"""Fused BF16 RMSNorm forward with FP32 statistics.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, H]`` input of any layout.  Row-strided views with a contiguous
        feature dimension, 32-byte aligned storage and a row stride that is a
        multiple of 32 bytes are read in place; other layouts (odd column offset,
        odd row stride, transposed storage) are materialized once.  A tensor of
        more dimensions, ``[..., H]``, is normalized row by row when its feature
        dimension is contiguous and its leading dimensions are contiguous with
        one another in row-major order (``stride(i) == stride(i + 1) * size(i +
        1)``; a ``[T, heads, H]`` slice of a wider ``[T, heads, W]`` buffer
        qualifies): the leading dimensions collapse into ``prod(...)`` rows, the
        outputs take the input's leading shape (``rstd`` is ``[...]``), ``w`` is
        shared by every row and the backward's ``dw`` sums them all; any other
        N-D layout raises ``ValueError`` naming the rule.  With ``residual`` this
        is the residual stream ``h`` and the normalized value is
        ``h_new = BF16(h + residual)``.
    w : torch.Tensor
        BF16 or FP32 ``[H]`` weight.  BF16 is read in place when contiguous and
        32-byte aligned (the forward kernels load it with 32-byte vectors), FP32
        -- the effective weight ``1 + w`` a zero-centred model computes in FP32
        -- when contiguous and 32-byte aligned; any other layout is copied once.
        An FP32 weight needs the FP32-weight route of the width on the device
        (:func:`cake_rmsnorm_det_train_weight_dtypes`), else ``ValueError``.
    eps : float
        Variance epsilon (FP32 inside the kernel).
    residual : torch.Tensor, optional
        BF16 update of ``x``'s shape added to ``x`` before normalization (same
        layout rules as ``x``).  Enables the fused residual-add route.
    out : torch.Tensor, optional
        Preallocated BF16 ``y`` of ``x``'s shape. Contiguous aligned buffers
        are written directly; any other layout receives a copy after the launch.
    rstd : torch.Tensor, optional
        Preallocated FP32 reciprocal RMS of ``x``'s leading shape (``[...]``
        for ``x`` of shape ``[..., H]``). Contiguous aligned buffers are written
        directly; any other layout receives a copy after the launch.
    residual_out : torch.Tensor, optional
        Preallocated BF16 ``h_new`` of ``x``'s shape, used only with ``residual``.
        Contiguous aligned buffers are written directly; any other layout
        receives a copy after the launch.

    Returns
    -------
    ``(y, rstd)`` or ``(y, rstd, h_new)`` when ``residual`` is given.
    ``rstd`` is the FP32 per-row ``1 / sqrt(mean(h_new^2) + eps)`` the backward
    needs; no other ``[T, H]`` temporary is produced.

    Notes
    -----
    The program windows of every architecture and route, and the program
    each window runs, are listed in the docstring of
    :func:`flashinfer.experimental.cake.cake_rmsnorm_det_train.cake_rmsnorm_det_train_forward`.
    """

    return _backend().cake_rmsnorm_det_train_forward(
        x, w, eps, residual=residual, out=out, rstd=rstd, residual_out=residual_out
    )


@flashinfer_api
def cake_rmsnorm_det_train_backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    *,
    g_residual: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fused BF16 RMSNorm backward: ``dx`` and the FP32 weight gradient ``dw``.

    Parameters
    ----------
    g : torch.Tensor
        BF16 ``[T, H]`` gradient of ``y`` of any layout.  Views with a contiguous
        feature dimension, 16-byte aligned storage and a row stride that is a
        multiple of 16 bytes (expanded or sliced autograd gradients included) are
        read in place; other layouts are materialized once.  An N-D gradient,
        ``[..., H]``, takes ``x``'s shape and collapses its leading dimensions
        into rows under the forward's rule.
    x : torch.Tensor
        The BF16 ``[T, H]`` (or ``[..., H]``) tensor that was normalized (``x`` of
        the plain forward, ``h_new`` of the residual forward); same layout rules
        as the forward's ``x``.
    w : torch.Tensor
        BF16 or FP32 ``[H]`` weight (the forward's).  BF16 is read in place when
        contiguous and 16-byte aligned, FP32 when contiguous and 32-byte aligned;
        any other layout is copied once.  An FP32 weight needs the FP32-weight
        route of the width (:func:`cake_rmsnorm_det_train_weight_dtypes`), else
        ``ValueError``.
    rstd : torch.Tensor
        FP32 ``[T]`` reciprocal RMS returned by the forward (of the leading
        shape ``[...]`` for an N-D call, or flat ``[prod(...)]``).
    g_residual : torch.Tensor, optional
        BF16 gradient of ``g``'s shape flowing directly through ``h_new``
        (residual forward), any layout.  It is added once so that the returned
        ``dx`` is the gradient of both residual-add inputs.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_rmsnorm_det_train_backward_workspace` sized for at least this
        many rows; allocated when omitted.
    dx : torch.Tensor, optional
        Preallocated BF16 gradient of ``x``'s shape. Contiguous aligned buffers
        are written directly; any other layout receives a copy after the launch.
    dw : torch.Tensor, optional
        Preallocated FP32 ``[H]`` weight gradient. Contiguous aligned buffers
        are written directly; any other layout receives a copy after the launch.

    Returns
    -------
    ``(dx, dw)`` with ``dw`` in FP32; the caller decides whether to cast it.
    For a fixed token count ``dw`` is bitwise reproducible on devices with the
    same SM count: the rows are split over ``ctas_per_sm`` x SM count CTAs, so
    devices of one architecture with different SM counts can give different
    ``dw`` bits.

    Notes
    -----
    The program windows of every architecture and route, and the program
    each window runs, are listed in the docstring of
    :func:`flashinfer.experimental.cake.cake_rmsnorm_det_train.cake_rmsnorm_det_train_backward`.
    """

    return _backend().cake_rmsnorm_det_train_backward(
        g,
        x,
        w,
        rstd,
        g_residual=g_residual,
        workspace=workspace,
        dx=dx,
        dw=dw,
    )


@flashinfer_api
def cake_rmsnorm_det(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
):
    """Autograd-enabled fused BF16 RMSNorm (optionally with a fused residual add).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, H]`` input on a supported CUDA device, or ``[..., H]`` under
        the layout rule of :func:`cake_rmsnorm_det_train_forward` (its leading
        dimensions collapse into rows).
    w : torch.Tensor
        BF16 or FP32 ``[H]`` RMSNorm weight (FP32 on the routes that ship the
        FP32-weight programs, see :func:`cake_rmsnorm_det_train_weight_dtypes`);
        the weight gradient is returned in ``w.dtype``.
    eps : float
        Variance epsilon evaluated in FP32 by the kernel.
    residual : torch.Tensor, optional
        BF16 update of ``x``'s shape added to ``x`` before normalization.  When
        provided, the normalized input ``h_new`` is returned with the output.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        BF16 ``y`` of ``x``'s shape, or ``(y, h_new)`` when ``residual``
        is given.  Backward launches the fused ``dx`` and deterministic ``dw``
        kernels of :func:`cake_rmsnorm_det_train_backward`.
    """

    return _backend().cake_rmsnorm_det(x, w, eps, residual=residual)


# The backend's own objects this module exports (the autograd class, the constants and
# the helpers) resolve on first access and are cached in this module; each is the
# backend's object itself.
_BACKEND_NAMES = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormDetFunction",
    "cake_rmsnorm_det_train_backward_workspace",
    "cake_rmsnorm_det_train_backward_workspace_bytes",
    "cake_rmsnorm_det_train_program_windows",
    "cake_rmsnorm_det_train_route_key",
    "cake_rmsnorm_det_train_weight_dtypes",
    "is_cake_rmsnorm_det_train_supported",
)


def __getattr__(name: str) -> Any:
    if name not in _BACKEND_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_backend(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_BACKEND_NAMES))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822
    "CakeRMSNormDetFunction",  # noqa: F822
    "cake_rmsnorm_det",
    "cake_rmsnorm_det_train_backward",
    "cake_rmsnorm_det_train_backward_workspace",  # noqa: F822
    "cake_rmsnorm_det_train_backward_workspace_bytes",  # noqa: F822
    "cake_rmsnorm_det_train_forward",
    "cake_rmsnorm_det_train_program_windows",  # noqa: F822
    "cake_rmsnorm_det_train_route_key",  # noqa: F822
    "cake_rmsnorm_det_train_weight_dtypes",  # noqa: F822
    "is_cake_rmsnorm_det_train_supported",  # noqa: F822
]
