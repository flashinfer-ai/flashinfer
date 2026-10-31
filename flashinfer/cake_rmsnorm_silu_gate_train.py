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

Cake SiLU-gated RMSNorm training forward / backward (SM100 / SM103 / SM107)
with a deterministic default and an opt-in non-deterministic weight gradient.

Forward, per row of 128 of ``x`` and the matching row of the separate gate
input ``z`` (``[T, H, 128]`` or ``[T * H, 128]``, BF16; ``weight`` ``[128]``
shared by every token and head)::

    rstd = 1 / sqrt(mean(x^2) + eps)                     # FP32
    y    = BF16(((x * rstd) * weight) * silu(z))         # FP32, one rounding

The backward recomputes the statistics and the gate from ``x`` and ``z`` and
returns BF16 ``dx`` and ``dz`` and the weight gradient summed over every token
and head, rounded once into BF16 or FP32.  The autograd function saves only its
inputs, which keeps ``torch.utils.checkpoint(..., use_reentrant=False)`` free of
FP32 temporaries.  No call synchronizes the host or the device; ``T`` and ``H``
are runtime values.

Every entry point takes the keyword ``deterministic`` (default ``True``):

* ``deterministic=True`` runs :mod:`flashinfer.cake_rmsnorm_silu_gate_det_train`
  with the same arguments: the weight gradient is accumulated in FP32 per CTA
  chunk and in FP64 over the chunks in an order fixed by the row count (a main
  kernel and a reduction kernel), so repeated calls return the same bits.  The
  default call returns exactly that package's results.
* ``deterministic=False`` runs this package's kernels: the same forward kernel
  (it has no reduction, so ``y`` has the same bits) and a one-kernel backward
  whose ``dx`` and ``dz`` are the deterministic backward's bits and whose CTAs
  add their FP64 weight-gradient column sums into FP64 accumulators with
  atomics; the CTA that arrives last rounds the sum once into the weight
  gradient.  The order of those FP64 additions depends on scheduling, so the
  weight gradient can differ by one rounding of its dtype between calls; the
  error bound is the deterministic backward's.
* ``deterministic=False`` while ``torch.use_deterministic_algorithms(True)`` is
  in effect raises ``RuntimeError``; with ``warn_only=True`` it warns and runs
  the deterministic kernels.  A call that asks for determinism never runs the
  non-deterministic kernels.

The non-deterministic backward's workspace
(``cake_rmsnorm_silu_gate_train_backward_workspace(T * H, device,
deterministic=False)``, a few KiB whatever ``T * H``) holds the FP64
accumulators and the arrival counter.  The kernel leaves it zero: allocate it
once, reuse it across calls, shapes, both weight-gradient dtypes and CUDA Graph
replays, and give each stream that may run a backward at the same time its own
buffer.  It is not interchangeable with the deterministic package's partial
buffer.

Usage::

    y = cake_rmsnorm_silu_gate(x, z, weight, eps)                         # deterministic
    y = cake_rmsnorm_silu_gate(x, z, weight, eps, deterministic=False)

    y = cake_rmsnorm_silu_gate_train_forward(x, z, weight, eps)
    dx, dz, dw = cake_rmsnorm_silu_gate_train_backward(
        dy, x, z, weight, eps, deterministic=False
    )

    # FP32 main-grad accumulation and a caller-owned workspace
    ws = cake_rmsnorm_silu_gate_train_backward_workspace(
        T * H, x.device, deterministic=False
    )
    cake_rmsnorm_silu_gate_train_backward(
        dy, x, z, weight, eps, grad_weight=main_grad, accumulate_grad_weight=True,
        workspace=ws, deterministic=False,
    )
"""

from __future__ import annotations

from typing import Optional

import torch

from . import cake_rmsnorm_silu_gate_train_backend as _backend
from .api_logging import flashinfer_api
from .cake_rmsnorm_silu_gate_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeRMSNormSiLUGateFunction,
    cake_rmsnorm_silu_gate_train_backward_workspace,
    cake_rmsnorm_silu_gate_train_backward_workspace_bytes,
    is_cake_rmsnorm_silu_gate_train_supported,
)

# The stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
def cake_rmsnorm_silu_gate_train_forward(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """See :func:`flashinfer.cake_rmsnorm_silu_gate_train_backend.cake_rmsnorm_silu_gate_train_forward`."""

    return _backend.cake_rmsnorm_silu_gate_train_forward(
        x, z, weight, eps, out=out, deterministic=deterministic
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_train_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    grad_x: Optional[torch.Tensor] = None,
    grad_z: Optional[torch.Tensor] = None,
    grad_weight: Optional[torch.Tensor] = None,
    accumulate_grad_weight: bool = False,
    compute_grad_weight: bool = True,
    workspace: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """See :func:`flashinfer.cake_rmsnorm_silu_gate_train_backend.cake_rmsnorm_silu_gate_train_backward`."""

    return _backend.cake_rmsnorm_silu_gate_train_backward(
        grad_out,
        x,
        z,
        weight,
        eps,
        grad_x=grad_x,
        grad_z=grad_z,
        grad_weight=grad_weight,
        accumulate_grad_weight=accumulate_grad_weight,
        compute_grad_weight=compute_grad_weight,
        workspace=workspace,
        deterministic=deterministic,
    )


@flashinfer_api
def cake_rmsnorm_silu_gate(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """See :func:`flashinfer.cake_rmsnorm_silu_gate_train_backend.cake_rmsnorm_silu_gate`."""

    return _backend.cake_rmsnorm_silu_gate(
        x, z, weight, eps, deterministic=deterministic
    )


# Sphinx renders the backend docstrings; the one-line delegations satisfy the checker.
cake_rmsnorm_silu_gate_train_forward.__doc__ = (
    _backend.cake_rmsnorm_silu_gate_train_forward.__doc__
)
cake_rmsnorm_silu_gate_train_backward.__doc__ = (
    _backend.cake_rmsnorm_silu_gate_train_backward.__doc__
)
cake_rmsnorm_silu_gate.__doc__ = _backend.cake_rmsnorm_silu_gate.__doc__


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormSiLUGateFunction",
    "cake_rmsnorm_silu_gate",
    "cake_rmsnorm_silu_gate_train_backward",
    "cake_rmsnorm_silu_gate_train_backward_workspace",
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_train_forward",
    "is_cake_rmsnorm_silu_gate_train_supported",
]
