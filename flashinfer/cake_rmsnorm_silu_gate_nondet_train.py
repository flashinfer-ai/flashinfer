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
of the non-deterministic family (``nondet``).  Every entry point runs only the
kernels of this family; the deterministic package is
:mod:`flashinfer.cake_rmsnorm_silu_gate_det_train`.

Forward, per row of 128 of ``x`` and the matching row of the separate gate
input ``z`` (``[T, H, 128]`` or ``[T * H, 128]``, BF16; ``weight`` ``[128]``
shared by every token and head)::

    rstd = 1 / sqrt(mean(x^2) + eps)                     # FP32
    y    = BF16(((x * rstd) * weight) * silu(z))         # FP32, one rounding

The backward recomputes the statistics and the gate from ``x`` and ``z`` and
returns BF16 ``dx`` and ``dz`` and the weight gradient summed over every token
and head, rounded once to FP32 (and that to BF16 for a BF16 weight gradient).
The autograd function saves only its inputs, which keeps
``torch.utils.checkpoint(..., use_reentrant=False)`` free of FP32 temporaries.
No call synchronizes the host or the device; ``T`` and ``H`` are runtime
values.

Every output repeats bit for bit: the same inputs on the same device give the
same ``y``, ``dx``, ``dz`` and weight gradient on every call, whatever the
grid, the order in which CTAs run and the work beside the call on other
streams.  The kernels are those of the standalone package
``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``:

- The forward computes ``rstd`` with ``rsqrt.approx`` and the SiLU gate with
  ``ex2.approx`` of ``-|z| log2(e)`` and ``rcp.approx`` (no flush of a nonzero
  sigmoid of a normal ``z``), every row's arithmetic fixed; its bits are not
  the deterministic package's.  Up to 12 rows per SM on SM100 and SM103 (59 on
  SM107) a forward with 16 threads to a row runs, and from 976 rows per SM on
  SM100 and SM103 (800 on SM107) a forward whose CTAs each walk several row
  blocks, with the same row arithmetic.
- The backward is one kernel over the deterministic backward's chunks of rows
  (8-row blocks, ``max(1, ceil(blocks / 8192))`` blocks a chunk): ``dx`` and
  ``dz`` are the deterministic backward's arithmetic, each chunk's FP32
  weight-gradient partial is the deterministic backward's, and the partials
  are added exactly, as signed 64-bit integer windows of their FP32 values (an
  addition of integers does not depend on its order); the CTA that arrives
  last rounds the exact sum once into the weight gradient.  So the weight
  gradient is the deterministic package's bits except where the exact sum lies
  within the deterministic FP64 sum's error of a rounding boundary, and an
  infinite or ``NaN`` partial gives its column the class the FP64 sum gives
  it.  While every chunk has a CTA of one wave (8 CTAs per SM) CTA ``b`` runs
  chunk ``b``; past that the CTAs claim runs of chunks from the workspace's
  claim counter, so a CTA that starts late takes fewer.  Up to 256 rows a
  BF16 weight gradient comes from one thread-block cluster of 16 CTAs with
  the same row arithmetic, which adds the CTAs' FP32 partials of each column
  in a fixed order in FP64, rounds once and uses no workspace.

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
3. Under ``torch.use_deterministic_algorithms(True)`` no call of this module
   alerts: every output repeats bit for bit.
4. Autograd: ``CakeRMSNormSiLUGateDetFunction`` and
   ``CakeRMSNormSiLUGateNondetFunction`` save only ``x``, ``z`` and ``weight``
   and run the backward of their own package.  A backward workspace serves
   only its own package (FP32 for the deterministic package, ``torch.uint8``
   for the non-deterministic one): each backward refuses the other package's
   workspace (``ValueError``), also in a call with no rows.
5. The former names are gone, with no alias or shim: the former modules fail
   at import, and the former names without a mode word are attributes of
   neither ``flashinfer`` nor either module.  The SiLU-gated RMSNorm section
   of ``docs/api/norm.rst`` maps each former name to its names in both modes.

The non-deterministic backward's workspace
(``cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, device)``, a
zero-filled ``torch.uint8`` tensor of 29192 bytes whatever ``T * H``) holds the
integer windows of the exact sum, 128 FP32 class words, the arrival ticket and
the claim counter; every launch of that backward uses one, also with
``compute_grad_weight=False``, but the one-cluster backward of up to 256 rows,
which reads none.  The kernel leaves it zero: allocate it once, reuse it
across calls, shapes, both weight-gradient dtypes and CUDA Graph replays
without a reset, and give each stream that may run a backward at the same
time its own buffer.  A buffer that did not come from that factory, or that
anything else wrote into, must be zeroed with
``cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)`` before its next
backward.  It is not interchangeable with the deterministic package's FP32
buffer: each backward refuses the other package's buffer (``ValueError``).
Calls without a workspace use one buffer per device, stream and backward
kernel (two per weight-gradient dtype: one for the one-wave backward, one for
the claimed backward), kept for later calls; the autograd node retains nothing
else.  They also keep a reserve of 64 zero-filled buffers per device (about
1.8 MiB, filled outside any capture): a call during CUDA Graph capture takes
one per capture and capture stream from it and keeps it for the life of the
process, so a replay launches no fill kernel (with none ready, the graph
zero-fills a buffer of its own).

Beside kernels resident on other streams (FlashInfer issue #7548): an SM
serves a CTA only once it can switch to a shared-memory carveout that holds
the CTA's dynamic shared memory, so a kernel on another stream whose CTAs hold
every SM with a small carveout (for example a long-running kernel with little
or no shared memory and the default carveout) can delay the start of a kernel
of this module until it exits.  Setting
``cudaFuncAttributePreferredSharedMemoryCarveout`` to
``cudaSharedmemCarveoutMaxShared`` on such a resident kernel, or giving it a
large dynamic shared-memory allocation, lets the backward start beside it.

Usage::

    y = cake_rmsnorm_silu_gate_nondet(x, z, weight, eps)                 # autograd

    y = cake_rmsnorm_silu_gate_nondet_train_forward(x, z, weight, eps)
    dx, dz, dw = cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, weight, eps)

    # FP32 main-grad accumulation and a caller-owned workspace
    ws = cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, x.device)
    cake_rmsnorm_silu_gate_nondet_train_backward(
        dy, x, z, weight, eps, grad_weight=main_grad, accumulate_grad_weight=True,
        workspace=ws,
    )
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api


# The backend package is imported on first use (flashinfer/experimental/README.md,
# "Containment rules"), so ``import flashinfer`` loads no experimental module.
@functools.cache
def _backend() -> Any:
    from .experimental.cake import cake_rmsnorm_silu_gate_nondet_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_rmsnorm_silu_gate_nondet_train


# The backend package's own objects (the autograd class and the architecture
# constants), resolved on first access by the module ``__getattr__`` below.
_BACKEND_OBJECTS = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormSiLUGateNondetFunction",
)

# The stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); the kernels and host bodies live in the
# self-contained backend package next to its generated CUDA sources.  The
# autograd function is the backend package's class.


@flashinfer_api
def is_cake_rmsnorm_silu_gate_nondet_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device with every route of the
    non-deterministic kernels.

    The deterministic package's query is
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.is_cake_rmsnorm_silu_gate_det_train_supported`.
    The answer does not depend on ``torch.use_deterministic_algorithms``.

    Parameters
    ----------
    device : torch.device
        The device to check.

    Returns
    -------
    bool
        ``True`` when the non-deterministic package has kernels for ``device``.
    """

    return _backend().is_cake_rmsnorm_silu_gate_nondet_train_supported(device)


@flashinfer_api
def cake_rmsnorm_silu_gate_nondet_train_forward(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""SiLU-gated RMSNorm forward: ``y = BF16(((x * rstd) * weight) * silu(z))``.

    ``rstd = 1 / sqrt(mean(x^2) + eps)`` over the last dimension (128), in FP32;
    ``silu(z) = z / (1 + exp(-z))`` in FP32; one rounding to BF16 at the end.
    The non-deterministic package's forward program: ``rstd`` from
    ``rsqrt.approx`` and the gate from ``ex2.approx`` and ``rcp.approx``, every
    row's arithmetic fixed, so ``y`` repeats bit for bit (its bits are not the
    deterministic package's; that one is
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_forward`).
    Up to 12 rows per SM on SM100 / SM103 (59 on SM107) a forward with 16
    threads to a row runs, and from 976 rows per SM on SM100 / SM103 (800 on
    SM107) a forward whose CTAs each walk several row blocks, with the same row
    arithmetic.  The forward does not alert under
    ``torch.use_deterministic_algorithms(True)``.

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

    return _backend().cake_rmsnorm_silu_gate_nondet_train_forward(
        x, z, weight, eps, out=out
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(
    rows: int, device: torch.device
) -> int:
    """Bytes of the caller-owned workspace of the non-deterministic backward
    for ``rows`` rows (``T * H``) on ``device``: the integer windows of the
    exact weight-gradient sum, 128 FP32 class words, the arrival ticket and the
    claim counter, about 29 KiB whatever ``rows``.

    The deterministic package's FP32 buffer has
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes`.

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

    return _backend().cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(
        rows, device
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_nondet_train_backward_workspace(
    rows: int, device: torch.device
) -> torch.Tensor:
    """A caller-owned workspace of the non-deterministic backward for ``rows``
    rows (``T * H``): a zero-filled ``torch.uint8`` tensor of
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes.

    The non-deterministic backward leaves it zero after every call, so it
    serves every later call on its stream (any ``T`` and ``H``, BF16 or FP32
    weight gradient) and every CUDA Graph replay without a reset.  A backward
    that may run at the same time on another stream needs its own workspace.
    It is not the deterministic package's FP32 buffer
    (:func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward_workspace`):
    each backward refuses the other package's buffer.

    Parameters
    ----------
    rows : int
        Number of rows of the backward call, ``T * H``.
    device : torch.device
        The CUDA device the backward runs on.

    Returns
    -------
    torch.Tensor
        The zero-filled ``torch.uint8`` workspace on ``device``.
    """

    return _backend().cake_rmsnorm_silu_gate_nondet_train_backward_workspace(
        rows, device
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_nondet_train_reset_workspace(
    workspace: torch.Tensor,
) -> None:
    """Return a caller-owned workspace of the non-deterministic backward to its
    first-use state.

    Zeroes the integer windows, the class words, the arrival ticket and the
    claim counter (a contiguous ``torch.uint8`` CUDA tensor of at least
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes
    starting at a 16-byte aligned address; all of it is zeroed) with one fill on the current stream of its device, without a
    host synchronization.  The non-deterministic backward leaves its workspace
    zero, so reuse across calls, shapes, weight-gradient dtypes and CUDA Graph
    replays needs no reset; a buffer that did not come from
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`, or that anything
    else wrote into (a launch that did not complete, another use of the
    buffer), needs one before its next backward.  Passing the deterministic
    package's FP32 buffer here is a ``ValueError``.  The reset does not depend
    on ``torch.use_deterministic_algorithms`` (a fill has
    no reduction).

    Parameters
    ----------
    workspace : torch.Tensor
        The caller-owned ``torch.uint8`` workspace of the non-deterministic
        backward.
    """

    _backend().cake_rmsnorm_silu_gate_nondet_train_reset_workspace(workspace)


@flashinfer_api
def cake_rmsnorm_silu_gate_nondet_train_backward(
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

    in FP32, the statistics and the gate recomputed from ``x`` and ``z``.

    One kernel of the non-deterministic package over the deterministic
    backward's chunks of rows: ``dx`` and ``dz`` are the deterministic
    backward's arithmetic, each chunk's FP32 weight-gradient partial is the
    deterministic backward's, and the partials are added exactly as 64-bit
    integer windows; the CTA that arrives last rounds the exact sum once into
    ``dw`` (adding ``grad_weight``'s value first when accumulating).  Every
    output repeats bit for bit, whatever the grid and the order in which CTAs
    run; ``dw`` is the deterministic backward's bits except where the exact sum
    lies within that backward's FP64 sum error of a rounding boundary.  Up to
    256 rows a BF16 ``dw`` comes from one cluster of 16 CTAs that adds the CTAs'
    FP32 partials of a column in a fixed order in FP64 and uses no workspace.
    The deterministic backward is
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward`.
    No call alerts under ``torch.use_deterministic_algorithms(True)``.

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
        (the exact sum rounded to FP32, and to BF16 from there for a BF16
        tensor).  Without it a new BF16 tensor is returned.
    accumulate_grad_weight : bool
        Add the gradient to ``grad_weight``'s current value before the
        rounding (exactly; needs ``grad_weight``).  A call with no rows leaves
        the accumulator as it is.
    compute_grad_weight : bool
        ``False`` skips the weight-gradient reduction and returns ``None`` for
        it; the kernel then touches neither ``grad_weight`` nor the
        workspace's windows, but its CTAs still take the arrival ticket (and
        past one wave claim their chunks from the claim counter), so it uses a
        workspace too (and leaves it zero).
    workspace : torch.Tensor, optional
        Caller-owned zero-filled ``torch.uint8`` buffer from
        :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`, which the
        kernel leaves zero (zero it with
        :func:`cake_rmsnorm_silu_gate_nondet_train_reset_workspace` after anything
        else wrote into it); the deterministic package's FP32 buffer is a
        ``ValueError``, also in a call with no rows.  When omitted, the
        backward uses one zero-filled buffer per device, stream and backward
        kernel, kept for later calls (during CUDA Graph capture, one
        zero-filled buffer per capture and capture stream, from a reserve
        those calls keep).  One buffer per backward that
        may run at the same time on another stream.

    Returns
    -------
    ``(grad_x, grad_z, grad_weight)``.
    """

    return _backend().cake_rmsnorm_silu_gate_nondet_train_backward(
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
def cake_rmsnorm_silu_gate_nondet(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable ``RMSNorm(x) * weight * SiLU(z)`` over the last dimension
    (128) on the non-deterministic kernels: a :class:`CakeRMSNormSiLUGateNondetFunction`
    node (the backend package's class), which saves only ``x``, ``z`` and
    ``weight``.

    Every output of the node repeats bit for bit, and no direction alerts
    under ``torch.use_deterministic_algorithms(True)``.  A non-reentrant
    checkpoint replays the forward through this function (the same ``y``).  The deterministic
    one is :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det`.

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

    return _backend().cake_rmsnorm_silu_gate_nondet(x, z, weight, eps)


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
    "CakeRMSNormSiLUGateNondetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_rmsnorm_silu_gate_nondet",
    "cake_rmsnorm_silu_gate_nondet_train_backward",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_nondet_train_forward",
    "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
    "is_cake_rmsnorm_silu_gate_nondet_train_supported",
]
