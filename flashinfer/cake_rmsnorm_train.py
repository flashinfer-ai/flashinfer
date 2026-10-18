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

Forward: ``r_t = (mean_j x_tj^2 + eps)^(-1/2)`` in FP32, ``y = BF16((x * r) * w)``.
The only tensor saved for the backward besides the inputs is the FP32 per-row
reciprocal RMS ``rstd`` (``[T]``).  Backward: one fused pass produces
``dx = BF16(r * g * w - x * r^3 * mean(g * w * x))`` and the FP32 weight
gradient ``dw = sum_t g * x * r`` through a fixed row-chunk partition and a
fixed-order reduction, so ``dw`` is bitwise reproducible for a given token
count on a given architecture.  The optional residual-add fusion computes
``h_new = BF16(h + u)`` first, normalizes ``h_new``, and in the backward adds
the gradient flowing directly through ``h_new`` so that one ``dx`` tensor is
the gradient of both ``h`` and ``u``.

All kernels use FP32 statistics and accumulation with IEEE division and
square root (no fast-math).  Inputs and outputs are BF16; ``rstd`` and ``dw``
are FP32.  Row-strided inputs (``x``, ``residual``, ``g``, ``g_residual``)
are accepted when ``stride(1) == 1``, ``stride(0)`` is a multiple of eight
elements and the storage is aligned (32 bytes for the forward inputs, 16 for
the backward); outputs are contiguous.  The
token count ``T`` is a runtime scalar: it may change between calls, ``T == 1``
is supported and ``T == 0`` returns empty outputs without launching.

Usage::

    y, rstd = cake_rmsnorm_train_forward(x, w, eps)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd)          # dw is FP32 [H]

    # autograd (dw is returned in w.dtype; the kernel accumulates in FP32)
    y = cake_rmsnorm(x, w, eps)
    y, h_new = cake_rmsnorm(h, w, eps, residual=u)                 # fused add

    # caller-owned backward workspace (partials + self-resetting counters)
    ws = cake_rmsnorm_train_backward_workspace(T, H, x.device)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd, workspace=ws)
"""

from __future__ import annotations

from typing import Optional

import torch

from . import cake_rmsnorm_train_backend as _backend
from .api_logging import flashinfer_api
from .cake_rmsnorm_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeRMSNormFunction,
    cake_rmsnorm_train_backward_workspace,
    cake_rmsnorm_train_backward_workspace_bytes,
    is_cake_rmsnorm_train_supported,
    route_key,
)

# The three stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
def cake_rmsnorm_train_forward(
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
        odd row stride, transposed storage) are materialized once.  With
        ``residual`` this is the residual stream ``h`` and the normalized value is
        ``h_new = BF16(h + residual)``.
    w : torch.Tensor
        BF16 ``[H]`` weight (copied when not contiguous).
    eps : float
        Variance epsilon (FP32 inside the kernel).
    residual : torch.Tensor, optional
        BF16 ``[T, H]`` update added to ``x`` before normalization (same layout
        rules as ``x``).  Enables the fused residual-add route.
    out, rstd, residual_out : torch.Tensor, optional
        Preallocated outputs: BF16 ``[T, H]`` ``y``, FP32 ``[T]`` reciprocal RMS
        and (residual only) BF16 ``[T, H]`` ``h_new``.  Contiguous aligned buffers
        are written directly; any other layout receives a copy after the launch.

    Returns
    -------
    ``(y, rstd)`` or ``(y, rstd, h_new)`` when ``residual`` is given.
    ``rstd`` is the FP32 per-row ``1 / sqrt(mean(h_new^2) + eps)`` the backward
    needs; no other ``[T, H]`` temporary is produced.
    """

    return _backend.cake_rmsnorm_train_forward(
        x, w, eps, residual=residual, out=out, rstd=rstd, residual_out=residual_out
    )


@flashinfer_api
def cake_rmsnorm_train_backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    *,
    deterministic: bool = True,
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
        read in place; other layouts are materialized once.
    x : torch.Tensor
        The BF16 ``[T, H]`` tensor that was normalized (``x`` of the plain
        forward, ``h_new`` of the residual forward); same layout rules as ``g``.
    w : torch.Tensor
        BF16 ``[H]`` weight (copied when not contiguous).
    rstd : torch.Tensor
        FP32 ``[T]`` reciprocal RMS returned by the forward.
    deterministic : bool
        Only the deterministic fixed-order ``dw`` reduction is exported;
        ``False`` is rejected.
    g_residual : torch.Tensor, optional
        BF16 ``[T, H]`` gradient flowing directly through ``h_new`` (residual
        forward), any layout.  It is added once so that the returned ``dx`` is
        the gradient of both residual-add inputs.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_rmsnorm_train_backward_workspace`; allocated when omitted.
    dx, dw : torch.Tensor, optional
        Preallocated outputs: BF16 ``[T, H]`` and FP32 ``[H]``.  Contiguous
        aligned buffers are written directly; any other layout receives a copy
        after the launch.

    Returns
    -------
    ``(dx, dw)`` with ``dw`` in FP32; the caller decides whether to cast it.
    For a fixed token count and architecture ``dw`` is bitwise reproducible.
    """

    return _backend.cake_rmsnorm_train_backward(
        g,
        x,
        w,
        rstd,
        deterministic=deterministic,
        g_residual=g_residual,
        workspace=workspace,
        dx=dx,
        dw=dw,
    )


@flashinfer_api
def cake_rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    deterministic: bool = True,
):
    """Autograd-enabled fused BF16 RMSNorm (optionally with a fused residual add).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, H]`` input on a supported CUDA device.
    w : torch.Tensor
        BF16 ``[H]`` RMSNorm weight.
    eps : float
        Variance epsilon evaluated in FP32 by the kernel.
    residual : torch.Tensor, optional
        BF16 ``[T, H]`` update added to ``x`` before normalization.  When
        provided, the normalized input ``h_new`` is returned with the output.
    deterministic : bool
        Whether to use the deterministic fixed-order weight-gradient reduction.
        Only ``True`` is supported; ``False`` is rejected during backward.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        BF16 ``y`` with shape ``[T, H]``, or ``(y, h_new)`` when ``residual``
        is given.  Backward launches the fused ``dx`` and deterministic ``dw``
        kernels of :func:`cake_rmsnorm_train_backward`.
    """

    return _backend.cake_rmsnorm(
        x, w, eps, residual=residual, deterministic=deterministic
    )


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormFunction",
    "cake_rmsnorm",
    "cake_rmsnorm_train_backward",
    "cake_rmsnorm_train_backward_workspace",
    "cake_rmsnorm_train_backward_workspace_bytes",
    "cake_rmsnorm_train_forward",
    "is_cake_rmsnorm_train_supported",
    "route_key",
]
