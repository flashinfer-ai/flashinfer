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

Cake SiLU-gated RMSNorm training forward / backward (SM100 / SM103 / SM107).

Forward, per row of 128 of ``x`` and the matching row of the separate gate
input ``z`` (``[T, H, 128]`` or ``[T * H, 128]``, BF16; ``weight`` ``[128]``
shared by every token and head)::

    rstd = 1 / sqrt(mean(x^2) + eps)                     # FP32, approximate rsqrt
    y    = BF16(((x * rstd) * weight) * silu(z))         # FP32, one rounding

This is ``RMSNorm(x) * weight * SiLU(z)``, not :func:`flashinfer.norm.fused_rmsnorm_silu`
(``SiLU(RMSNorm(x))``).  The backward recomputes the statistics and the gate
from ``x`` and ``z`` and returns BF16 ``dx`` and ``dz`` and the weight
gradient summed over every token and head (FP32 runs of at most 16 rows, added
exactly in fixed point, rounded once to FP32), so repeated calls with the same row
count on a device with the same SM count return the same bits.  The autograd
function saves only its inputs, which keeps
``torch.utils.checkpoint(..., use_reentrant=False)`` free of FP32 temporaries.
No call synchronizes the host or the device; ``T`` and ``H`` are runtime values.

The forward computes ``rstd`` and the gate with the approximate FP32
reciprocal square root, base-2 exponential and reciprocal, so ``y`` can differ
from the correctly rounded FP32 formula by one BF16 unit in the last place
where that value lies within a few FP32 units in the last place of a BF16
rounding boundary; for finite ``z`` below about ``-87.3`` the gate is ``-0``.  The
backward computes them with the same approximate instructions but keeps the
subnormal sigmoid down to about ``-88.72``, so ``dx`` and ``dz`` lie within one
BF16 unit in the last place of the correctly rounded FP32 formulas plus 1/64 of
a unit of the row's largest term.  The forward's CTAs each walk at most two
blocks of 16 rows, and the backward's CTAs, past one wave, claim chunks of row
blocks from a counter, so a CTA that becomes resident late takes only the
blocks left.  No kernel waits for another CTA or grid.  Beside a kernel on
another stream that holds ``k`` of the ``N`` SMs, the forward takes at most
``N / (N - k)`` times its time alone, plus the share of one CTA's row blocks in
the call, plus 0.02 (a CTA whose slot that kernel holds walks all of its blocks
once it starts).

Mode and dispatch rules:

1. The mode is part of every name.  Every public name of the deterministic
   module :mod:`flashinfer.cake_rmsnorm_silu_gate_det_train` (backend package
   ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train``) carries
   ``det``; every public name of the non-deterministic module
   :mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train` (backend package
   ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``)
   carries ``nondet``.  The module constants ``ARCHES`` and
   ``SUPPORTED_COMPUTE_CAPABILITIES`` are exempt: they keep their names,
   without a mode word, in both modules and both backend packages.  The
   ``flashinfer`` top level exports only names with a mode word.
   Each module imports its backend package on first use (``import
   flashinfer`` loads neither backend package); its autograd class and
   its architecture constants are the backend package's objects.
2. No entry point takes a ``deterministic`` keyword: passing one is a
   ``TypeError``.  A call of one package never runs the other package.
3. Under ``torch.use_deterministic_algorithms(True)`` no deterministic call
   alerts: every deterministic launch gives the same bits for the same
   inputs on the same device (a sum across CTAs is exact).  A call of
   :mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train` alerts only when it
   is a backward that computes a BF16 weight gradient in the fused band
   (every architecture, up to 1000 rows per SM; see that module): it raises
   PyTorch's alert immediately before that launch (``RuntimeError``; with
   ``warn_only=True`` a ``UserWarning``, then the call runs), and no other
   call of that module alerts.  The check runs on the host: a CUDA Graph
   checks it when it is captured, not when it is replayed.
4. Autograd: ``CakeRMSNormSiLUGateDetFunction`` and
   ``CakeRMSNormSiLUGateNondetFunction`` save only ``x``, ``z`` and ``weight``
   and run the backward of their own package.  A backward workspace serves
   only its own package (the deterministic package's is FP32, the
   non-deterministic package's is not): each backward refuses the other
   package's workspace (``ValueError``), also in a call with no rows.
5. The former names are gone, with no alias or shim: the former modules fail
   at import, and the former names without a mode word are attributes of
   neither ``flashinfer`` nor either module.  The SiLU-gated RMSNorm section
   of ``docs/api/norm.rst`` maps each former name to its names in both modes.

Usage::

    y = cake_rmsnorm_silu_gate_det(x, z, weight, eps)                    # autograd

    y = cake_rmsnorm_silu_gate_det_train_forward(x, z, weight, eps)
    dx, dz, dw = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, weight, eps)

    # FP32 main-grad accumulation and a caller-owned workspace (zero-filled;
    # every backward leaves it zero, so it serves every later call on its stream)
    ws = cake_rmsnorm_silu_gate_det_train_backward_workspace(T * H, x.device)
    cake_rmsnorm_silu_gate_det_train_backward(
        dy, x, z, weight, eps, grad_weight=main_grad, accumulate_grad_weight=True,
        workspace=ws,
    )
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api

# The entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.  The backend package is
# imported on first use (flashinfer/experimental/README.md, "Containment
# rules"), so ``import flashinfer`` loads no experimental module.


@functools.cache
def _backend() -> Any:
    from .experimental.cake import cake_rmsnorm_silu_gate_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_rmsnorm_silu_gate_det_train


# The backend package's own objects (the autograd class and the architecture
# constants), resolved on first access by the module ``__getattr__`` below.
_BACKEND_OBJECTS = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormSiLUGateDetFunction",
)


@flashinfer_api
def is_cake_rmsnorm_silu_gate_det_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of an exported architecture.

    Parameters
    ----------
    device : torch.device
        The device to check.

    Returns
    -------
    bool
        ``True`` when the deterministic package has kernels for ``device``.
    """

    return _backend().is_cake_rmsnorm_silu_gate_det_train_supported(device)


@flashinfer_api
def cake_rmsnorm_silu_gate_det_train_forward(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""SiLU-gated RMSNorm forward: ``y = BF16(((x * rstd) * weight) * silu(z))``.

    ``rstd = 1 / sqrt(mean(x^2) + eps)`` over the last dimension (128) and
    ``silu(z) = z / (1 + exp(-z))``, in FP32 with the approximate reciprocal
    square root, base-2 exponential and reciprocal; one rounding to BF16 at the
    end.  The same inputs give the same bits at every call.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[..., 128]`` input (``[T, H, 128]`` or ``[T * H, 128]``).
    z : torch.Tensor
        BF16 gate input of the same shape as ``x`` (its own strides).
    weight : torch.Tensor
        BF16 ``[128]`` weight shared by every token and head.
    eps : float
        Variance epsilon (FP32 inside the kernel).
    out : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape (``out`` when given).
    """

    return _backend().cake_rmsnorm_silu_gate_det_train_forward(
        x, z, weight, eps, out=out
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
    rows: int, device: torch.device
) -> int:
    """Bytes of the caller-owned backward workspace for ``rows`` rows
    (``T * H``) on ``device``: the weight gradient's fixed-point digit
    counters, one flag word a column, the arrival ticket and the claim counter,
    18952 bytes whatever ``rows``.

    Parameters
    ----------
    rows : int
        Number of rows of the backward call, ``T * H``.
    device : torch.device
        The CUDA device the backward runs on.

    Returns
    -------
    int
        The workspace size in bytes.
    """

    return _backend().cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
        rows, device
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_det_train_backward_workspace(
    rows: int, device: torch.device
) -> torch.Tensor:
    """A caller-owned backward workspace for ``rows`` rows: a zero-filled FP32
    tensor of :func:`cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes`
    bytes.  The backward leaves it zero after every call, so it serves every
    later call on its stream (any ``T`` and ``H``, BF16 or FP32 weight
    gradient) and every CUDA Graph replay without a reset; a backward that may
    run at the same time on another stream needs its own workspace.

    Parameters
    ----------
    rows : int
        Number of rows of the backward call, ``T * H``.
    device : torch.device
        The CUDA device the backward runs on.

    Returns
    -------
    torch.Tensor
        The zero-filled FP32 workspace on ``device``.
    """

    return _backend().cake_rmsnorm_silu_gate_det_train_backward_workspace(rows, device)


@flashinfer_api
def cake_rmsnorm_silu_gate_det_train_reset_workspace(workspace: torch.Tensor) -> None:
    """Return a caller-owned backward workspace to its first-use state (all
    zero).

    Every backward leaves its workspace zero, so reuse across calls, shapes,
    weight-gradient dtypes and CUDA Graph replays needs no reset.  A workspace
    that did not come from
    :func:`cake_rmsnorm_silu_gate_det_train_backward_workspace`, or that
    something else wrote into (a launch that did not complete, another use of
    the buffer), must be reset before its next backward.  The fill runs on the
    current stream without a host synchronization (it can be captured in a CUDA
    Graph).

    Parameters
    ----------
    workspace : torch.Tensor
        A contiguous, 16-byte aligned FP32 CUDA tensor of at least
        :func:`cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes`
        bytes; a workspace of the non-deterministic package is refused
        (``ValueError``).
    """

    _backend().cake_rmsnorm_silu_gate_det_train_reset_workspace(workspace)


@flashinfer_api
def cake_rmsnorm_silu_gate_det_train_backward(
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
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""SiLU-gated RMSNorm backward: ``dx``, ``dz`` and the weight gradient.

    With ``n = x * rstd``, ``s = sigmoid(z)`` and ``a = dy * silu(z)``::

        dz = BF16(dy * (n * weight) * s * (1 + z * (1 - s)))
        dx = BF16(rstd * a * weight - x * rstd^3 * mean(a * weight * x))
        dw = sum over every token and head of a * n

    in FP32, the statistics and the gate recomputed from ``x`` and ``z`` with
    the approximate reciprocal square root, base-2 exponential and reciprocal:
    ``dx`` and ``dz`` lie within one BF16 unit in the last place of these
    formulas in correctly rounded FP32 plus 1/64 of a unit of the row's
    largest term, the same bits at every call.
    ``dw`` is summed in FP32 runs of at most 16 rows, added exactly in fixed
    point whatever the order of the CTAs, and rounded once to FP32 (then to BF16
    for a BF16 gradient): the same row count on a device with the same SM count
    gives the same bits.

    Parameters
    ----------
    grad_out, x, z : torch.Tensor
        BF16 tensors of one ``[..., 128]`` shape (any strides, see the module).
    weight : torch.Tensor
        BF16 ``[128]``.
    eps : float
        The forward's epsilon.
    grad_x, grad_z : torch.Tensor, optional
        Preallocated BF16 outputs of ``x``'s shape.
    grad_weight : torch.Tensor, optional
        A BF16 or FP32 ``[128]`` tensor that receives the weight gradient
        (the exact sum rounded once to FP32, then to BF16 for a BF16 tensor).
        Without it a new BF16 tensor is returned.
    accumulate_grad_weight : bool
        Add the gradient to ``grad_weight``'s current value exactly, before
        the rounding (needs ``grad_weight``).
    compute_grad_weight : bool
        ``False`` skips the weight-gradient result and returns ``None`` for it.
    workspace : torch.Tensor, optional
        Zero-filled FP32 workspace of at least
        :func:`cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes` bytes,
        contiguous and 16-byte aligned, left zero by the call; the stream's
        own workspace when omitted.  A workspace of the non-deterministic
        package is refused (``ValueError``), also in a call with no rows.

    Returns
    -------
    ``(grad_x, grad_z, grad_weight)``.
    """

    return _backend().cake_rmsnorm_silu_gate_det_train_backward(
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
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_det(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Differentiable ``RMSNorm(x) * weight * SiLU(z)`` over the last dimension
    (128): :class:`CakeRMSNormSiLUGateDetFunction` applied to the arguments.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[..., 128]`` input (``[T, H, 128]`` or ``[T * H, 128]``).
    z : torch.Tensor
        BF16 gate input of the same shape as ``x``.
    weight : torch.Tensor
        BF16 ``[128]`` weight shared by every token and head.
    eps : float
        Variance epsilon (FP32 inside the kernel).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.
    """

    return _backend().cake_rmsnorm_silu_gate_det(x, z, weight, eps)


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        value = getattr(_backend(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_BACKEND_OBJECTS))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822 - resolved by __getattr__
    "CakeRMSNormSiLUGateDetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_rmsnorm_silu_gate_det",
    "cake_rmsnorm_silu_gate_det_train_backward",
    "cake_rmsnorm_silu_gate_det_train_backward_workspace",
    "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_det_train_forward",
    "cake_rmsnorm_silu_gate_det_train_reset_workspace",
    "is_cake_rmsnorm_silu_gate_det_train_supported",
]
