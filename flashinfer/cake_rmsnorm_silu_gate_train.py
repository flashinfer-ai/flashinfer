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
and head, rounded once into BF16 or FP32.  The autograd functions save only
their inputs, which keeps ``torch.utils.checkpoint(..., use_reentrant=False)``
free of FP32 temporaries.  No call synchronizes the host or the device; ``T``
and ``H`` are runtime values.

Every entry point takes the keyword ``deterministic`` (default ``True``) and
dispatches on it here:

* ``deterministic=True`` runs the deterministic package
  :mod:`flashinfer.cake_rmsnorm_silu_gate_det_train` with the same arguments:
  the weight gradient is accumulated in FP32 per CTA chunk and in FP64 over the
  chunks in an order fixed by the row count (a main kernel and a reduction
  kernel), so repeated calls return the same bits.  The default call returns
  exactly that package's results and raises its errors; the autograd entry
  :func:`cake_rmsnorm_silu_gate` builds its ``CakeRMSNormSiLUGateDetFunction``
  node.
* ``deterministic=False`` runs the standalone non-deterministic package
  ``flashinfer.cake_rmsnorm_silu_gate_train_backend``: the same forward kernel
  (it has no reduction, so ``y`` has the same bits; on SM100 and SM103 up to
  10 rows per SM a forward with 16 threads to a row and from 976 rows per SM
  a forward whose CTAs each walk several row blocks, also the same bits) and
  a one-kernel backward whose ``dx`` and ``dz`` are the
  deterministic backward's bits.  Its CTAs take row blocks (one block each
  when the grid has a CTA for every block;
  past one wave of CTAs they claim the blocks in chunks), sum the weight
  gradient in FP32 runs of at most 16 rows folded into FP64 (a CTA of a few
  blocks adds its runs across its warps in FP32 and converts once: at most 16
  FP32 roundings a term either way), and add their FP64 column sums into FP64
  accumulators with atomics; the CTA that arrives last
  rounds the sum once into the weight gradient.  Which blocks each CTA claims,
  and so the rows of each FP32 run, and the order of the FP64 additions depend
  on scheduling: the weight gradient can differ between calls, always within
  the deterministic backward's error bound ``|dw64| * 2^-7 + sum|terms| *
  2^-20`` per column (``dw64`` the exact sum of the ``terms``).  That
  package's entry points take ``deterministic=False`` only (any other value is
  a ``ValueError``).
* ``deterministic=False`` while ``torch.use_deterministic_algorithms(True)`` is
  in effect raises ``RuntimeError``; with ``warn_only=True`` it warns and runs
  the deterministic package.  A call that asks for determinism never runs the
  non-deterministic kernels.  The support query and the workspace reset do not
  depend on ``torch.use_deterministic_algorithms``.  A
  ``cake_rmsnorm_silu_gate(..., deterministic=False)`` node whose backward runs
  after the deterministic algorithms were enabled raises PyTorch's alert for an
  operation without a deterministic implementation (``RuntimeError``; under
  ``warn_only=True`` a ``UserWarning``, and the non-deterministic backward
  runs) when ``weight`` needs a gradient.

``deterministic`` must be a ``bool``: any other value (``None``, ``0``, a
string) is a ``TypeError`` and selects neither package.
:class:`CakeRMSNormSiLUGateFunction` takes it as its last forward input
(default ``True``) and runs the selected package in both directions.

The non-deterministic backward's workspace
(``cake_rmsnorm_silu_gate_train_backward_workspace(T * H, device,
deterministic=False)``, 4104 bytes whatever ``T * H``) holds the FP64
accumulators, the arrival ticket and the claim counter the backward's CTAs
claim row blocks from when the rows have more blocks than one wave of CTAs;
every launch of that backward uses one, also with
``compute_grad_weight=False``.  The kernel leaves it zero: allocate
it once, reuse it across calls, shapes, both weight-gradient dtypes and CUDA
Graph replays without a reset, and give each stream that may run a backward at
the same time its own buffer.  A buffer that did not come from that factory, or
that anything else wrote into, must be zeroed with
``cake_rmsnorm_silu_gate_train_reset_workspace(ws, deterministic=False)``
before its next backward.  It is not interchangeable with the deterministic
package's partial buffer.  Calls without a workspace use one 4104-byte buffer
per device, stream and backward kernel (one per weight-gradient dtype, two
from 1024 rows per SM on SM107 and from 1280 on SM100 and SM103), kept for
later calls; the
autograd node retains nothing else.  They also keep a reserve of 64
zero-filled 4104-byte buffers per device (about 272 KiB, filled outside any
capture): a call during CUDA Graph capture takes one per capture and capture
stream from it and keeps it for the life of the process, so a replay launches
no fill kernel (with none ready, the graph zero-fills a buffer of its own).

Beside kernels resident on other streams (FlashInfer issue #7548; observed on
B200, sm_100a): each CTA of the non-deterministic backward needs 20736 bytes of
dynamic shared memory (an SM of sm_100a holds four such CTAs), and an SM serves
it only once it can switch to a shared-memory carveout that holds it.  A kernel
on another stream whose CTAs hold every SM with a small carveout (for example a
long-running kernel with little or no shared memory and the default carveout)
therefore delays that backward's start until the kernel exits.  Setting
``cudaFuncAttributePreferredSharedMemoryCarveout`` to
``cudaSharedmemCarveoutMaxShared`` on such a resident kernel, or giving it a
large dynamic shared-memory allocation, lets the backward start beside it.  The
deterministic backward also waits for such a kernel when that kernel has no
shared memory and the default or the maximum-L1 carveout; both forwards start
beside it whatever its carveout.  The non-deterministic backward launches at
most one wave of CTAs, or on SM100 and SM103 from 1280 rows per SM 5 CTAs per
SM, a quarter more than an SM holds at once.  When the grid has a CTA for every
row block, CTA ``c`` takes block ``c``; past one wave the CTAs claim every
block in chunks from the workspace's claim counter, each CTA its next chunk as
the last block of its current one starts, so a CTA that starts late takes
fewer blocks (none when every chunk was claimed before it started).

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

import warnings
from typing import Optional

import torch

from . import cake_rmsnorm_silu_gate_det_train_backend as _deterministic
from . import cake_rmsnorm_silu_gate_train_backend as _backend
from .api_logging import flashinfer_api
from .cake_rmsnorm_silu_gate_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
)

# The deterministic entry point each entry point of this module runs for
# ``deterministic=True``.
_DETERMINISTIC_NAMES = {
    "cake_rmsnorm_silu_gate_train_forward": "cake_rmsnorm_silu_gate_det_train_forward",
    "cake_rmsnorm_silu_gate_train_backward": "cake_rmsnorm_silu_gate_det_train_backward",
    "cake_rmsnorm_silu_gate_train_backward_workspace": "cake_rmsnorm_silu_gate_det_train_backward_workspace",
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes": "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate": "cake_rmsnorm_silu_gate_det",
    "CakeRMSNormSiLUGateFunction": "CakeRMSNormSiLUGateDetFunction",
}


def _check_mode(deterministic: bool, api: str) -> None:
    """``deterministic`` is a ``bool``; any other value is a ``TypeError``."""

    if not isinstance(deterministic, bool):
        raise TypeError(
            f"flashinfer.{api}: deterministic must be a bool, got "
            f"{type(deterministic).__name__}"
        )


def _resolve_deterministic(deterministic: bool, api: str) -> bool:
    """Whether a call of ``api`` runs the deterministic package.

    ``deterministic=True`` does.  ``deterministic=False`` runs the
    non-deterministic package unless ``torch.use_deterministic_algorithms(True)``
    is in effect: then the call raises ``RuntimeError``, or, with
    ``warn_only=True``, warns and runs the deterministic package.  A value that
    is not a ``bool`` is a ``TypeError``.
    """

    _check_mode(deterministic, api)
    if deterministic:
        return True
    if not torch.are_deterministic_algorithms_enabled():
        return False
    message = (
        f"flashinfer.{api}(deterministic=False) selects the kernels whose weight "
        "gradient is summed over the CTAs in a scheduling-dependent order, but "
        "torch.use_deterministic_algorithms(True) is in effect; call it with "
        "deterministic=True or use "
        f"flashinfer.cake_rmsnorm_silu_gate_det_train.{_DETERMINISTIC_NAMES[api]}"
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{message}; running the deterministic kernels", UserWarning, stacklevel=3
        )
        return True
    raise RuntimeError(message)


# The stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); the kernels and host bodies live in the
# two self-contained backend packages next to their generated CUDA sources.


@flashinfer_api
def is_cake_rmsnorm_silu_gate_train_supported(
    device: torch.device, *, deterministic: bool = True
) -> bool:
    """Whether ``device`` is a CUDA device of an exported architecture.

    ``deterministic=True`` answers for the deterministic package
    (:func:`flashinfer.cake_rmsnorm_silu_gate_det_train.is_cake_rmsnorm_silu_gate_det_train_supported`),
    ``False`` for the non-deterministic package's routes.  The answer does not
    depend on ``torch.use_deterministic_algorithms``.

    Parameters
    ----------
    device : torch.device
        The device to check.
    deterministic : bool
        ``True`` (default) answers for the deterministic package, ``False`` for
        the non-deterministic package.

    Returns
    -------
    bool
        ``True`` when the selected package has kernels for ``device``.
    """

    _check_mode(deterministic, "is_cake_rmsnorm_silu_gate_train_supported")
    if deterministic:
        return _deterministic.is_cake_rmsnorm_silu_gate_det_train_supported(device)
    return _backend.is_cake_rmsnorm_silu_gate_train_supported(
        device, deterministic=False
    )


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
    r"""SiLU-gated RMSNorm forward: ``y = BF16(((x * rstd) * weight) * silu(z))``.

    ``rstd = 1 / sqrt(mean(x^2) + eps)`` over the last dimension (128), in FP32;
    ``silu(z) = z / (1 + exp(-z))`` in FP32; one rounding to BF16 at the end.

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
    deterministic : bool
        ``True`` (default) runs the deterministic package's forward
        (:func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_forward`)
        with these arguments.  ``False`` runs the non-deterministic package's
        forward program, the same kernel rendered there, or on SM100 / SM103
        up to 10 rows per SM a forward with 16 threads to a row and from 976
        rows per SM a forward whose CTAs each walk several row blocks, with
        the same arithmetic (the forward has no reduction, so
        both return the same bits); it is refused while
        ``torch.use_deterministic_algorithms(True)`` is in effect (with
        ``warn_only=True``: a warning, then the deterministic forward).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape (``out`` when given).
    """

    if _resolve_deterministic(deterministic, "cake_rmsnorm_silu_gate_train_forward"):
        return _deterministic.cake_rmsnorm_silu_gate_det_train_forward(
            x, z, weight, eps, out=out
        )
    return _backend.cake_rmsnorm_silu_gate_train_forward(
        x, z, weight, eps, out=out, deterministic=False
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_train_backward_workspace_bytes(
    rows: int, device: torch.device, *, deterministic: bool = True
) -> int:
    """Bytes of the caller-owned backward workspace for ``rows`` rows
    (``T * H``) on ``device``.

    ``deterministic`` is resolved as the backward resolves it (see
    :func:`cake_rmsnorm_silu_gate_train_backward`), so the size fits the kernel
    the backward call with the same flag runs.  ``True``: the deterministic
    package's FP32 partial buffer, which grows with ``rows``
    (:func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes`).
    ``False``: the FP64 weight-gradient accumulators and the 8-byte word of
    the arrival ticket and the claim counter, a few KiB whatever ``rows``.

    Parameters
    ----------
    rows : int
        Number of rows of the backward call, ``T * H``.
    device : torch.device
        The CUDA device the backward runs on.
    deterministic : bool
        The ``deterministic`` flag of the backward call the workspace is for.

    Returns
    -------
    int
        The workspace size in bytes.
    """

    if _resolve_deterministic(
        deterministic, "cake_rmsnorm_silu_gate_train_backward_workspace_bytes"
    ):
        return _deterministic.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
            rows, device
        )
    return _backend.cake_rmsnorm_silu_gate_train_backward_workspace_bytes(
        rows, device, deterministic=False
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_train_backward_workspace(
    rows: int, device: torch.device, *, deterministic: bool = True
) -> torch.Tensor:
    """A caller-owned backward workspace for ``rows`` rows (``T * H``).

    ``True``: the deterministic package's uninitialized FP32 partial buffer
    (:func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward_workspace`).
    ``False``: a zero-filled FP64 tensor of
    :func:`cake_rmsnorm_silu_gate_train_backward_workspace_bytes` bytes.  The
    non-deterministic backward leaves it zero after every call, so it serves
    every later call on its stream (any ``T`` and ``H``, BF16 or FP32 weight
    gradient) and every CUDA Graph replay without a reset.  A backward that may
    run at the same time on another stream needs its own workspace.  The two
    kinds do not substitute for one another: the deterministic buffer holds
    nonzero partials after a call, and the non-deterministic backward refuses a
    buffer that is not FP64.

    Parameters
    ----------
    rows : int
        Number of rows of the backward call, ``T * H``.
    device : torch.device
        The CUDA device the backward runs on.
    deterministic : bool
        The ``deterministic`` flag of the backward call the workspace is for.

    Returns
    -------
    torch.Tensor
        The workspace on ``device``: FP32 for ``True``, zero-filled FP64 for
        ``False``.
    """

    if _resolve_deterministic(
        deterministic, "cake_rmsnorm_silu_gate_train_backward_workspace"
    ):
        return _deterministic.cake_rmsnorm_silu_gate_det_train_backward_workspace(
            rows, device
        )
    return _backend.cake_rmsnorm_silu_gate_train_backward_workspace(
        rows, device, deterministic=False
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_train_reset_workspace(
    workspace: torch.Tensor, *, deterministic: bool = True
) -> None:
    """Return a caller-owned backward workspace to its first-use state.

    ``False``: zero the FP64 accumulators, the arrival ticket and the claim
    counter of a non-deterministic workspace (a contiguous FP64 CUDA tensor of
    at least :func:`cake_rmsnorm_silu_gate_train_backward_workspace_bytes`
    bytes; all of it is zeroed) with one fill on the current stream of its
    device, without a host synchronization.  The non-deterministic backward
    leaves its workspace zero, so reuse across calls, shapes, weight-gradient
    dtypes and CUDA Graph replays needs no reset; a buffer that did not come from
    :func:`cake_rmsnorm_silu_gate_train_backward_workspace`, or that anything
    else wrote into (a launch that did not complete, another use of the buffer),
    needs one before its next backward.  ``True``: the deterministic backward
    writes every element of its FP32 partial buffer before reading it, so that
    buffer holds no state between calls; the call checks that ``workspace`` is
    an FP32 tensor (an FP64 non-deterministic workspace is a ``ValueError``,
    never silently skipped) and launches nothing.  Neither form depends on
    ``torch.use_deterministic_algorithms`` (a fill has no reduction).

    Parameters
    ----------
    workspace : torch.Tensor
        The caller-owned backward workspace.
    deterministic : bool
        ``False`` for a non-deterministic FP64 workspace, ``True`` (default)
        for a deterministic FP32 one.
    """

    _check_mode(deterministic, "cake_rmsnorm_silu_gate_train_reset_workspace")
    if deterministic:
        if not isinstance(workspace, torch.Tensor):
            raise TypeError("workspace must be a torch.Tensor")
        if workspace.dtype != torch.float32:
            raise ValueError(
                "deterministic=True takes the deterministic backward's FP32 "
                "workspace, which needs no reset; reset the FP64 workspace of the "
                "non-deterministic backward with deterministic=False"
            )
        return
    _backend.cake_rmsnorm_silu_gate_train_reset_workspace(
        workspace, deterministic=False
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
    r"""SiLU-gated RMSNorm backward: ``dx``, ``dz`` and the weight gradient.

    With ``n = x * rstd``, ``s = sigmoid(z)`` and ``a = dy * silu(z)``::

        dz = BF16(dy * (n * weight) * s * (1 + z * (1 - s)))
        dx = BF16(rstd * a * weight - x * rstd^3 * mean(a * weight * x))
        dw = sum over every token and head of a * n

    in FP32, the statistics and the gate recomputed from ``x`` and ``z``.

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
        (rounded once into its dtype).  Without it a new BF16 tensor is returned.
    accumulate_grad_weight : bool
        Add the gradient to ``grad_weight``'s current value before the one
        rounding (needs ``grad_weight``).  A call with no rows leaves the
        accumulator as it is.
    compute_grad_weight : bool
        ``False`` skips the weight-gradient reduction and returns ``None`` for
        it; the non-deterministic kernel then touches neither ``grad_weight``
        nor the workspace's accumulators, but its CTAs still take the arrival
        ticket (and past one wave of row blocks claim their blocks from the
        claim counter), so it uses a workspace too (and leaves it zero).
    workspace : torch.Tensor, optional
        Caller-owned buffer from :func:`cake_rmsnorm_silu_gate_train_backward_workspace`
        called with the same ``deterministic`` flag: the deterministic
        package's FP32 partial buffer (sized for at least this many rows), or the
        zero-filled FP64 buffer of the non-deterministic backward, which the
        kernel leaves zero (zero it with
        :func:`cake_rmsnorm_silu_gate_train_reset_workspace` after anything else
        wrote into it).  When omitted: the deterministic package allocates its
        buffer per call; the non-deterministic backward uses one zero-filled
        buffer per device, stream and backward kernel, kept for later
        calls (during CUDA Graph capture, one zero-filled buffer per capture
        and capture stream, from a reserve those calls keep).  One buffer per
        backward that may run at the same time on another stream.
    deterministic : bool
        ``True`` (default) runs the deterministic package's backward
        (:func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward`)
        with these arguments: ``dw`` accumulated in FP32 per CTA chunk and in
        FP64 over the chunks in an order fixed by the row count, rounded once,
        so repeated calls return the same bits.  ``False`` runs one kernel of
        the non-deterministic package: the same ``dx`` and ``dz`` bits, and each
        CTA sums the rows of the blocks it takes (its own block when the grid
        has a CTA for every block, else the chunks it claims) in FP32 runs of
        at most 16 rows folded into FP64 (a CTA of a few blocks adds its runs
        across its warps in FP32 and converts once: at most 16 FP32 roundings
        a term either way) and adds its FP64 column sums into FP64
        accumulators with atomics; the CTA that
        arrives last rounds the sum once into ``dw`` (adding ``grad_weight``'s
        value first when accumulating).  The claimed blocks, and so the FP32
        runs, and the order of the FP64 additions depend on scheduling: ``dw``
        can differ between calls, always within the deterministic backward's
        error bound ``|dw64| * 2^-7 + sum|terms| * 2^-20`` per column.
        ``False`` is refused with ``RuntimeError`` while
        ``torch.use_deterministic_algorithms(True)`` is in effect; with
        ``warn_only=True`` it warns and runs the deterministic backward (which
        then allocates its own buffer in place of a non-deterministic
        ``workspace``).

    Returns
    -------
    ``(grad_x, grad_z, grad_weight)``.
    """

    if _resolve_deterministic(deterministic, "cake_rmsnorm_silu_gate_train_backward"):
        if (
            not deterministic
            and isinstance(workspace, torch.Tensor)
            and workspace.dtype == torch.float64
        ):
            # A deterministic=False call that fell back under warn_only: its
            # FP64 workspace is not the deterministic backward's buffer.
            workspace = None
        return _deterministic.cake_rmsnorm_silu_gate_det_train_backward(
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
        deterministic=False,
    )


class CakeRMSNormSiLUGateFunction(torch.autograd.Function):
    """Autograd of ``RMSNorm(x) * weight * SiLU(z)`` with the numerical mode as
    an input: saves ``x``, ``z`` and ``weight`` (no FP32 intermediate) and
    recomputes the rest in the backward, so
    ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only the inputs as
    saved state.

    ``forward(x, z, weight, eps, deterministic=True)`` resolves
    ``deterministic`` once, as :func:`cake_rmsnorm_silu_gate` does, and both
    directions run the selected package: ``True`` (the default) the
    deterministic package's forward and backward with the arguments
    ``CakeRMSNormSiLUGateDetFunction`` passes (its bits and errors), ``False``
    the non-deterministic package's entry points with ``deterministic=False``
    (the backward raises PyTorch's alert if
    ``torch.use_deterministic_algorithms(True)`` is in effect when it runs and
    ``weight`` needs a gradient).  :func:`cake_rmsnorm_silu_gate` builds the
    selected package's own node instead.
    """

    @staticmethod
    def forward(ctx, x, z, weight, eps, deterministic=True):
        ctx.deterministic = _resolve_deterministic(
            deterministic, "CakeRMSNormSiLUGateFunction"
        )
        if ctx.deterministic:
            y = _deterministic.cake_rmsnorm_silu_gate_det_train_forward(
                x, z, weight, eps
            )
        else:
            y = _backend.cake_rmsnorm_silu_gate_train_forward(
                x, z, weight, eps, deterministic=False
            )
        ctx.save_for_backward(x, z, weight)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, z, weight = ctx.saved_tensors
        if ctx.deterministic:
            dx, dz, dw = _deterministic.cake_rmsnorm_silu_gate_det_train_backward(
                grad_y,
                x,
                z,
                weight,
                ctx.eps,
                compute_grad_weight=ctx.needs_input_grad[2],
            )
        else:
            dx, dz, dw = _backend.cake_rmsnorm_silu_gate_train_backward(
                grad_y,
                x,
                z,
                weight,
                ctx.eps,
                compute_grad_weight=ctx.needs_input_grad[2],
                deterministic=False,
            )
        return (
            dx if ctx.needs_input_grad[0] else None,
            dz if ctx.needs_input_grad[1] else None,
            dw,
            None,
            None,
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
    """Differentiable ``RMSNorm(x) * weight * SiLU(z)`` over the last dimension
    (128).  Each autograd node saves only ``x``, ``z`` and ``weight``.

    ``deterministic=True`` (default) is
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det`
    (a ``CakeRMSNormSiLUGateDetFunction`` node): the deterministic package in
    both directions, its bits.  ``False`` applies the non-deterministic
    package's own ``CakeRMSNormSiLUGateFunction``
    (``flashinfer.cake_rmsnorm_silu_gate_train_backend``), whose weight gradient
    is summed over the CTAs in a scheduling-dependent order; it is refused while
    ``torch.use_deterministic_algorithms(True)`` is in effect (with
    ``warn_only=True``: a warning, then the deterministic node).  The backward
    of a ``False`` node raises PyTorch's alert if the deterministic algorithms
    are enabled by the time it runs and ``weight`` needs a gradient
    (``RuntimeError``; under ``warn_only=True`` a ``UserWarning``, then the
    non-deterministic backward).  A non-reentrant checkpoint replays the forward
    through this function with the same flag.

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
    deterministic : bool
        ``True`` (default) applies the deterministic node, ``False`` the
        non-deterministic one (see above).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.
    """

    if _resolve_deterministic(deterministic, "cake_rmsnorm_silu_gate"):
        return _deterministic.cake_rmsnorm_silu_gate_det(x, z, weight, eps)
    return _backend.cake_rmsnorm_silu_gate(x, z, weight, eps, deterministic=False)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormSiLUGateFunction",
    "cake_rmsnorm_silu_gate",
    "cake_rmsnorm_silu_gate_train_backward",
    "cake_rmsnorm_silu_gate_train_backward_workspace",
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_train_forward",
    "cake_rmsnorm_silu_gate_train_reset_workspace",
    "is_cake_rmsnorm_silu_gate_train_supported",
]
