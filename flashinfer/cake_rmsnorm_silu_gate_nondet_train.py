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
with the non-deterministic weight gradient (``nondet``).  Every entry point
runs only the non-deterministic kernels; the deterministic package is
:mod:`flashinfer.cake_rmsnorm_silu_gate_det_train`.

Forward, per row of 128 of ``x`` and the matching row of the separate gate
input ``z`` (``[T, H, 128]`` or ``[T * H, 128]``, BF16; ``weight`` ``[128]``
shared by every token and head)::

    rstd = 1 / sqrt(mean(x^2) + eps)                     # FP32
    y    = BF16(((x * rstd) * weight) * silu(z))         # FP32, one rounding

The backward recomputes the statistics and the gate from ``x`` and ``z`` and
returns BF16 ``dx`` and ``dz`` and the weight gradient summed over every token
and head, its FP64 sum rounded to FP32 (and that to BF16 for a BF16 weight
gradient: FP64 -> FP32 -> BF16).  The autograd function saves only its
inputs, which keeps ``torch.utils.checkpoint(..., use_reentrant=False)`` free
of FP32 temporaries.  No call synchronizes the host or the device; ``T``
and ``H`` are runtime values.

The kernels are those of the standalone non-deterministic package
``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``: the deterministic
package's forward kernel (it has no reduction, so ``y`` has the same bits; up
to 12 rows per SM on SM100 and SM103, 59 on SM107, a forward with 16 threads
to a row, and from 976 rows per SM on SM100 and SM103, 800 on SM107, a forward
whose CTAs each walk several row blocks, also the same bits) and a one-kernel
backward whose ``dx`` and ``dz`` are the deterministic backward's bits.  Its
CTAs take row blocks (one block each when the grid has a CTA for every block;
past one wave of CTAs they claim the blocks in chunks), sum the weight
gradient in FP32 runs of at most 16 rows folded into FP64 (a CTA of a few
blocks adds its runs across its warps in FP32 and converts once: at most 16
FP32 roundings a term either way), and add their FP64 column sums into FP64
accumulators with atomics; the CTA that arrives last
rounds the sum into the weight gradient (FP64 -> FP32 -> BF16 for BF16).
Which blocks each CTA claims, and so the rows of each FP32 run, and the
order of the FP64 additions depend on scheduling: the weight gradient can
differ between calls, always within the deterministic backward's error
bound ``|dw64| * 2^-7 + sum|terms| * 2^-20`` per column (``dw64`` the exact
sum of the ``terms``) when the gate ``silu(z)``, the terms and the column
sums are FP32 normal numbers or zero.  FP32 underflows gradually: a
subnormal gate, term or sum is rounded to a multiple of ``2^-149``, an
error that a later product scales (by ``|dy|`` for the gate, by
``|x * rstd| <= sqrt(128)`` for ``dy * silu(z)``), so with subnormal values
the weight gradient of either package can lie outside that bound (by up to
about ``(1 + |x * rstd|) * 2^-150`` per row when only ``dy * silu(z)`` and
the terms underflow), and a weight gradient below the smallest normal
number of its dtype is rounded to a multiple of the dtype's smallest
subnormal (``2^-133`` for BF16, ``2^-149`` for FP32).  An infinite
``grad_out`` makes its columns' weight gradient infinite or ``NaN``:
``NaN`` where the infinite term is in the last row and the last row block
is partial (its padding rows reread that row and multiply it by zero), as
in the deterministic backward.  On SM100 from 800 rows per SM and on SM103
from 700, up to 12000 (from 12000 the backward above), each thread keeps
its folded FP32 runs,
and each CTA its column sums over its row groups, as FP32 two-sum pairs
(a sum and its error term) instead of FP64, and converts a column's pair
to FP64 once for the atomic add.  The error that fold itself adds to a
CTA's column sum ``S`` of its FP32 runs is at most
``C(k) * 2^-48 * A + 2^-53 * |S|`` (``k`` the folds of a thread, ``A``
the sum of the runs' magnitudes, ``C(k) = (k - 1)(k - 2) + 16 (k + 14)``;
the FP64 fold's is ``(k + 14) * 2^-53 * A``): a bound on the fold alone,
not on the weight gradient's error, which the bound above states.  A
column whose pair sum is not finite takes the FP64 value of the sum
alone, so an infinite or ``NaN`` term gives the column the class the FP64
fold gives it.  With finite terms whose FP32 pair sum in a thread or in a
CTA's column passes the FP32 range, the weight gradient can be ``+-inf``
or ``NaN`` where the deterministic backward returns a finite value.  Up to
256 rows the backward is one
thread-block cluster of 16 CTAs with the same row arithmetic, which adds
the CTAs' FP32 partials of each column in a fixed order in FP64 and rounds
the sum the same way: there the weight gradient is the same bits on every
call, within the same bound, and no workspace is used.

Mode and dispatch rules (the same in
:mod:`flashinfer.cake_rmsnorm_silu_gate_det_train` and
:mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train`):

1. The mode is part of every name.  Every public name of the deterministic
   module :mod:`flashinfer.cake_rmsnorm_silu_gate_det_train` (backend package
   ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train``) carries
   ``det``; every public name of the non-deterministic module
   :mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train` (backend package
   ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``)
   carries ``nondet``.  The ``flashinfer`` top level exports only such names.
2. No entry point takes a ``deterministic`` keyword: passing one is a
   ``TypeError``.  A call of one package never runs the other package.
3. Under ``torch.use_deterministic_algorithms(True)`` no deterministic call
   alerts: every deterministic launch sums in a fixed order.  The
   non-deterministic forward never alerts.  A non-deterministic backward that
   computes the weight gradient raises PyTorch's alert immediately before its
   order-varying launch (``RuntimeError``; with ``warn_only=True`` a
   ``UserWarning``, then the non-deterministic kernel runs).  The one-cluster
   backward of up to 256 rows (a fixed order), a call with no rows and a call
   with ``compute_grad_weight=False`` do not alert.  The check runs on the
   host: a CUDA Graph checks it when it is captured, not when it is replayed.
4. Autograd: ``CakeRMSNormSiLUGateDetFunction`` and
   ``CakeRMSNormSiLUGateNondetFunction`` save only ``x``, ``z`` and ``weight``
   and run the backward of their own package.  A backward workspace holds the
   raw partial sums of one package and serves only that package (FP32 for the
   deterministic package, FP64 for the non-deterministic one): each backward
   refuses the other package's workspace (``ValueError``).
5. The former names are gone, with no alias or shim: the modules
   ``flashinfer.cake_rmsnorm_silu_gate_train``,
   ``flashinfer.cake_rmsnorm_silu_gate_train_backend`` and
   ``flashinfer.cake_rmsnorm_silu_gate_det_train_backend`` fail at import, and
   the names without a mode word (``CakeRMSNormSiLUGateFunction``,
   ``cake_rmsnorm_silu_gate``, ``cake_rmsnorm_silu_gate_train_forward``,
   ``cake_rmsnorm_silu_gate_train_backward``,
   ``cake_rmsnorm_silu_gate_train_backward_workspace``,
   ``cake_rmsnorm_silu_gate_train_backward_workspace_bytes``,
   ``cake_rmsnorm_silu_gate_train_reset_workspace`` and
   ``is_cake_rmsnorm_silu_gate_train_supported``) are attributes of neither
   ``flashinfer`` nor either module.

The non-deterministic backward's workspace
(``cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, device)``,
4104 bytes whatever ``T * H``) holds the FP64
accumulators, the arrival ticket and the claim counter the backward's CTAs
claim row blocks from when the rows have more blocks than one wave of CTAs;
every launch of that backward uses one, also with
``compute_grad_weight=False``, but the one-cluster backward of up to 256 rows,
which reads none.  The kernel leaves it zero: allocate
it once, reuse it across calls, shapes, both weight-gradient dtypes and CUDA
Graph replays without a reset, and give each stream that may run a backward at
the same time its own buffer.  A buffer that did not come from that factory, or
that anything else wrote into, must be zeroed with
``cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)``
before its next backward.  It is not interchangeable with the deterministic
package's FP32 partial buffer: each backward refuses the other package's
buffer (``ValueError``).  Calls without a workspace use one 4104-byte buffer
per device, stream and backward kernel (one per weight-gradient dtype, two
from 1024 rows per SM on SM107, two from 800 rows per SM on SM100 and from 700
on SM103, and three from 12000 on both), kept for later calls; the
autograd node retains nothing else.  They also keep a reserve of 64
zero-filled 4104-byte buffers per device (about 272 KiB, filled outside any
capture): a call during CUDA Graph capture takes one per capture and capture
stream from it and keeps it for the life of the process, so a replay launches
no fill kernel (with none ready, the graph zero-fills a buffer of its own).

Beside kernels resident on other streams (FlashInfer issue #7548; observed on
B200, sm_100a, with the 20736-byte CTAs and on R200, sm_107a, with the
20736-byte and the 10496-byte CTAs): above 256 rows, and on SM107 below 1024
rows per SM, each CTA of the non-deterministic backward
needs 20736 bytes of dynamic shared memory (an SM of sm_100a holds four such
CTAs; from 1024 rows per SM on SM107 each CTA of the long backward needs 10496
bytes), and an SM serves it only once it can switch to a shared-memory
carveout that holds it.  A kernel on another stream whose CTAs hold every SM
with a small carveout (for example a long-running kernel with little or no
shared memory and the default carveout) therefore delays that backward's start
until the kernel exits.  Setting
``cudaFuncAttributePreferredSharedMemoryCarveout`` to
``cudaSharedmemCarveoutMaxShared`` on such a resident kernel, or giving it a
large dynamic shared-memory allocation, lets the backward start beside it.  The
deterministic backward also waits for such a kernel when that kernel has no
shared memory and the default or the maximum-L1 carveout; both forwards start
beside it whatever its carveout.  On SM100 and SM103 the one-cluster backward
(up to 256 rows; 16 CTAs of 6144 bytes whatever the rows), measured at 48 and
240 rows, started at once beside a kernel holding every SM with one 128-thread
CTA, no shared memory and the default, the maximum-L1 or the maximum-shared
carveout (B200 and GB300).  On SM107 it waits for such a kernel with the default
or the maximum-L1 carveout, as the other backwards do, and starts beside it with
the maximum-shared carveout (R200).  The non-deterministic backward launches at
most one wave of CTAs, or on SM100 from 800 rows per SM and on SM103 from 700
5 CTAs per SM, a quarter more than an SM holds at once.  When the grid has a
CTA for every row block, CTA ``c`` takes block ``c``; past one wave the CTAs
claim every block
in chunks from the workspace's claim counter, each CTA its next chunk as the
last block of its current one starts, so a CTA that starts late takes fewer
blocks (none when every chunk was claimed before it started).

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

from typing import Optional

import torch

from .api_logging import flashinfer_api
from .experimental.cake import cake_rmsnorm_silu_gate_nondet_train as _backend
from .experimental.cake.cake_rmsnorm_silu_gate_nondet_train import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeRMSNormSiLUGateNondetFunction,
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

    return _backend.is_cake_rmsnorm_silu_gate_nondet_train_supported(device)


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
    The non-deterministic package's forward program: the deterministic
    package's forward kernel rendered there, or up to 12 rows per SM on SM100 /
    SM103 (59 on SM107) a forward with 16 threads to a row, and from 976 rows
    per SM on SM100 / SM103 (800 on SM107) a forward whose CTAs each walk
    several row blocks, with the same arithmetic (the forward has no reduction,
    so both packages return the same bits; the deterministic one is
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_forward`).
    The forward does not alert under
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

    return _backend.cake_rmsnorm_silu_gate_nondet_train_forward(
        x, z, weight, eps, out=out
    )


@flashinfer_api
def cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(
    rows: int, device: torch.device
) -> int:
    """Bytes of the caller-owned workspace of the non-deterministic backward
    for ``rows`` rows (``T * H``) on ``device``: the FP64 weight-gradient
    accumulators and the 8-byte word of the arrival ticket and the claim
    counter, a few KiB whatever ``rows``.

    The deterministic package's FP32 partial buffer, which grows with
    ``rows``, has
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

    return _backend.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(rows, device)


@flashinfer_api
def cake_rmsnorm_silu_gate_nondet_train_backward_workspace(
    rows: int, device: torch.device
) -> torch.Tensor:
    """A caller-owned workspace of the non-deterministic backward for ``rows``
    rows (``T * H``): a zero-filled FP64 tensor of
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes.

    The non-deterministic backward leaves it zero after every call, so it
    serves every later call on its stream (any ``T`` and ``H``, BF16 or FP32
    weight gradient) and every CUDA Graph replay without a reset.  A backward
    that may run at the same time on another stream needs its own workspace.
    It is not the deterministic package's FP32 partial buffer
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
        The zero-filled FP64 workspace on ``device``.
    """

    return _backend.cake_rmsnorm_silu_gate_nondet_train_backward_workspace(rows, device)


@flashinfer_api
def cake_rmsnorm_silu_gate_nondet_train_reset_workspace(workspace: torch.Tensor) -> None:
    """Return a caller-owned workspace of the non-deterministic backward to its
    first-use state.

    Zeroes the FP64 accumulators, the arrival ticket and the claim counter (a
    contiguous FP64 CUDA tensor of at least
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes; all of
    it is zeroed) with one fill on the current stream of its device, without a
    host synchronization.  The non-deterministic backward leaves its workspace
    zero, so reuse across calls, shapes, weight-gradient dtypes and CUDA Graph
    replays needs no reset; a buffer that did not come from
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`, or that anything
    else wrote into (a launch that did not complete, another use of the
    buffer), needs one before its next backward.  The deterministic package's
    FP32 partial buffer needs no reset (the deterministic backward writes every
    element before reading it); passing one here is a ``ValueError``.  The
    reset does not depend on ``torch.use_deterministic_algorithms`` (a fill has
    no reduction).

    Parameters
    ----------
    workspace : torch.Tensor
        The caller-owned FP64 workspace of the non-deterministic backward.
    """

    _backend.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(workspace)


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

    One kernel of the non-deterministic package: the
    same ``dx`` and ``dz`` bits as the deterministic backward, and each CTA
    sums the rows of the blocks it takes (its own block when the grid has a
    CTA for every block, else the chunks it claims) in FP32 runs of at most
    16 rows folded into FP64 (a CTA of a few blocks adds its runs across
    its warps in FP32 and converts once: at most 16 FP32 roundings a term
    either way) and adds its FP64 column sums into FP64 accumulators with
    atomics; the CTA that arrives last rounds the sum into ``dw`` (adding
    ``grad_weight``'s value first when accumulating).  The claimed blocks,
    and so the FP32 runs, and the order of the FP64 additions depend on
    scheduling: ``dw`` can differ between calls, always within the
    deterministic backward's error bound
    ``|dw64| * 2^-7 + sum|terms| * 2^-20`` per column while the gate, the
    terms and the sums are FP32 normal numbers or zero (subnormal values
    and infinities: see the module documentation).  On SM100 from 800 rows
    per SM and on SM103 from 700, up to 12000, the runs are folded, and a
    CTA's column sums kept, as FP32 two-sum pairs, converted to FP64 once
    per column (that fold's own error bound and its finite-overflow case:
    see the module documentation).  Up to 256 rows one cluster of 16 CTAs
    runs instead: the CTAs' FP32 partials of a column are added in a fixed
    order in FP64, so ``dw`` is the same bits on every call (within the
    same bound), and the workspace is not used.  The deterministic backward is
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward`
    (``dw`` accumulated in FP32 per CTA chunk and in FP64 over the chunks in an
    order fixed by the row count).
    Under ``torch.use_deterministic_algorithms(True)`` a call that computes
    the weight gradient raises PyTorch's alert immediately before the
    order-varying launch (``RuntimeError``; with ``warn_only=True`` a
    ``UserWarning``, then the non-deterministic kernel runs); the
    one-cluster backward of up to 256 rows, a call with no rows and a call
    with ``compute_grad_weight=False`` do not alert.

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
        (the FP64 sum rounded to FP32, and to BF16 from there for a BF16
        tensor).  Without it a new BF16 tensor is returned.
    accumulate_grad_weight : bool
        Add the gradient to ``grad_weight``'s current value before the
        rounding (in FP64; needs ``grad_weight``).  A call with no rows leaves
        the accumulator as it is.
    compute_grad_weight : bool
        ``False`` skips the weight-gradient reduction and returns ``None`` for
        it; the non-deterministic kernel then touches neither ``grad_weight``
        nor the workspace's accumulators, but its CTAs still take the arrival
        ticket (and past one wave of row blocks claim their blocks from the
        claim counter), so it uses a workspace too (and leaves it zero); the
        one-cluster backward of up to 256 rows uses none.
    workspace : torch.Tensor, optional
        Caller-owned zero-filled FP64 buffer from
        :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`, which the
        kernel leaves zero (zero it with
        :func:`cake_rmsnorm_silu_gate_nondet_train_reset_workspace` after anything
        else wrote into it); the deterministic package's FP32 buffer is a
        ``ValueError``.  When omitted, the backward uses one zero-filled buffer
        per device, stream and backward kernel, kept for later calls (during
        CUDA Graph capture, one zero-filled buffer per capture and capture
        stream, from a reserve those calls keep).  One buffer per backward that
        may run at the same time on another stream.

    Returns
    -------
    ``(grad_x, grad_z, grad_weight)``.
    """

    return _backend.cake_rmsnorm_silu_gate_nondet_train_backward(
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

    The backward of the node raises PyTorch's alert if the deterministic
    algorithms are enabled by the time it runs and ``weight`` needs a gradient
    on more than 256 rows (``RuntimeError``; under ``warn_only=True`` a
    ``UserWarning``, then the non-deterministic backward).  A non-reentrant
    checkpoint replays the forward through this function.  The deterministic
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

    return _backend.cake_rmsnorm_silu_gate_nondet(x, z, weight, eps)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormSiLUGateNondetFunction",
    "cake_rmsnorm_silu_gate_nondet",
    "cake_rmsnorm_silu_gate_nondet_train_backward",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_nondet_train_forward",
    "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
    "is_cake_rmsnorm_silu_gate_nondet_train_supported",
]
