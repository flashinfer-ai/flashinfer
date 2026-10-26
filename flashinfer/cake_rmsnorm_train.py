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
    """See :func:`flashinfer.cake_rmsnorm_train_backend.cake_rmsnorm_train_forward`."""

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
    """See :func:`flashinfer.cake_rmsnorm_train_backend.cake_rmsnorm_train_backward`."""

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
    """See :func:`flashinfer.cake_rmsnorm_train_backend.cake_rmsnorm`."""

    return _backend.cake_rmsnorm(
        x, w, eps, residual=residual, deterministic=deterministic
    )


# Sphinx renders the backend docstrings; the one-line delegations satisfy the checker.
cake_rmsnorm_train_forward.__doc__ = _backend.cake_rmsnorm_train_forward.__doc__
cake_rmsnorm_train_backward.__doc__ = _backend.cake_rmsnorm_train_backward.__doc__
cake_rmsnorm.__doc__ = _backend.cake_rmsnorm.__doc__


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
