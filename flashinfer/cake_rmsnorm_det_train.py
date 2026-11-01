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
are FP32.  ``w`` is BF16, or FP32 on the routes that ship the FP32-weight
programs (the effective weight ``1 + w`` of a zero-centred model, computed by
the caller in FP32; :func:`weight_dtypes` lists them per route).  Row-strided inputs (``x``, ``residual``, ``g``, ``g_residual``)
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
count" in the forward / backward docstrings and :func:`program_windows`).

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

- SM100 / SM103: the residual-fused H 6144 backward.
- SM107: the residual-fused H 6144 backward, the backward of H 1536 above 1024
  tokens, the backward of H 2048 above 256 tokens, the backward of H 4096
  above 1024 tokens and the forward of H 6144 above 4096 tokens.

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

from typing import Optional

import torch

from . import cake_rmsnorm_det_train_backend as _backend
from .api_logging import flashinfer_api
from .cake_rmsnorm_det_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeRMSNormDetFunction,
    cake_rmsnorm_det_train_backward_workspace,
    cake_rmsnorm_det_train_backward_workspace_bytes,
    is_cake_rmsnorm_det_train_supported,
    program_windows,
    route_key,
    weight_dtypes,
)

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
    """See :func:`flashinfer.cake_rmsnorm_det_train_backend.cake_rmsnorm_det_train_forward`."""

    return _backend.cake_rmsnorm_det_train_forward(
        x, w, eps, residual=residual, out=out, rstd=rstd, residual_out=residual_out
    )


@flashinfer_api
def cake_rmsnorm_det_train_backward(
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
    """See :func:`flashinfer.cake_rmsnorm_det_train_backend.cake_rmsnorm_det_train_backward`."""

    return _backend.cake_rmsnorm_det_train_backward(
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
def cake_rmsnorm_det(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    deterministic: bool = True,
):
    """See :func:`flashinfer.cake_rmsnorm_det_train_backend.cake_rmsnorm_det`."""

    return _backend.cake_rmsnorm_det(
        x, w, eps, residual=residual, deterministic=deterministic
    )


# Sphinx renders the backend docstrings; the one-line delegations satisfy the checker.
cake_rmsnorm_det_train_forward.__doc__ = _backend.cake_rmsnorm_det_train_forward.__doc__
cake_rmsnorm_det_train_backward.__doc__ = (
    _backend.cake_rmsnorm_det_train_backward.__doc__
)
cake_rmsnorm_det.__doc__ = _backend.cake_rmsnorm_det.__doc__


# The names before the det rename, the same objects as the det names of this facade (not in __all__).
CakeRMSNormFunction = CakeRMSNormDetFunction
cake_rmsnorm = cake_rmsnorm_det
cake_rmsnorm_train_backward = cake_rmsnorm_det_train_backward
cake_rmsnorm_train_backward_workspace = cake_rmsnorm_det_train_backward_workspace
cake_rmsnorm_train_backward_workspace_bytes = (
    cake_rmsnorm_det_train_backward_workspace_bytes
)
cake_rmsnorm_train_forward = cake_rmsnorm_det_train_forward
is_cake_rmsnorm_train_supported = is_cake_rmsnorm_det_train_supported


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormDetFunction",
    "cake_rmsnorm_det",
    "cake_rmsnorm_det_train_backward",
    "cake_rmsnorm_det_train_backward_workspace",
    "cake_rmsnorm_det_train_backward_workspace_bytes",
    "cake_rmsnorm_det_train_forward",
    "is_cake_rmsnorm_det_train_supported",
    "program_windows",
    "route_key",
    "weight_dtypes",
]
