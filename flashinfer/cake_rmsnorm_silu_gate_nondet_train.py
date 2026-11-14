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

``y``, ``dx``, ``dz`` and an FP32 weight gradient repeat bit for bit: the
same inputs on the same device give the same bits on every call, whatever the
order in which CTAs run and the work beside the call on other streams, and
whatever the grid.  A BF16 weight gradient repeats bit for bit too, whatever
the grid but in the repeatable bands (below: past 1000 rows per SM on SM100
and SM103; on SM107 past 512 rows up to 64 rows per SM and past 256; there
the rows and the SM count fix the grid and the bits), except in the fused
band (below: past 512 rows on SM100, past 256 rows on SM103, up to 1000 rows
per SM; on SM107 past 64 up to 256 rows per SM), where its last bits may vary
from run to run.  A band runs a
backward whose weight gradient may vary only where that makes the
checkpointed training step (2 x forward + backward) at least 3 % faster than
the repeatable backward.  The kernels are those of the standalone package
``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``:

- The forward computes ``rstd`` with ``rsqrt.approx`` and the SiLU gate with
  ``ex2.approx`` of ``-z log2(e)`` and ``rcp.approx`` of a scaled ``1 + e``
  (the deterministic forward's split gate: a nonzero sigmoid down to ``z``
  about -88.72, where ``ex2.approx`` overflows); its bits need not be the
  deterministic package's.
  Where ``x`` and ``z`` hold packed rows (row ``n`` at element ``128 n``) a
  forward with 16 threads to a row runs up to 38 rows per SM on SM100, 36
  on SM103 and 65 on SM107, other layouts up to 12 rows per SM on SM100 and
  SM103 (59 on SM107), and past that, in bands of rows per SM, the pipelined
  forwards (builds of the deterministic package's forward program with a
  gate that does not flush), forwards whose CTAs each walk several row
  blocks and, for other layouts past 700 rows per SM on SM100 and SM103 (480
  on SM107), a forward with 8 threads to a row and one CTA per 8-row block.
  Each forward program's arithmetic is fixed, but the programs
  associate a row's sum of squares differently (the 16-thread forwards:
  8-term FMA chains of 16 lanes folded by XOR shuffles 1, 2, 4, 8; the SM107
  pipelined forward: four 4-term chains of 8 lanes added in pairs, then
  folded by 1, 2, 4; the others: 16-term chains of 8 lanes folded by 1, 2, 4;
  the row count, the SM count and the layout pick the program), so ``y``'s bits can
  differ between bands while each program repeats them for the same inputs
  on the same device.
- The backward is one kernel; ``dx`` and ``dz`` are one row arithmetic in
  every band but the fused and the repeatable ones, within one BF16 unit in
  the last place of the FP64 value plus 1/64 of a unit of the tensor's largest
  finite magnitude (the deterministic package's backward need not give these
  bits; the fused and the repeatable bands give its bits).  A call of few rows
  runs one thread-block cluster of 16 CTAs, which adds the weight-gradient
  partials of each column in a fixed order in FP64, rounds that sum to FP32
  (then to BF16 for a BF16 weight gradient) and uses no workspace.  The short
  backward (256-thread CTAs, every CTA's FP32 partials added in rank order)
  runs a BF16 weight gradient or ``compute_grad_weight=False`` up to 256 rows;
  its weight gradient repeats its bits, which need not be the deterministic
  package's.  The exact tiny backward (the chunk partials of the deterministic
  backward at FlashInfer 2e989556 in its reduction order: that pair's bits)
  runs an FP32 weight gradient up to 512 rows on every architecture and a BF16
  one (or ``compute_grad_weight=False``) from 257 up to 512 rows on SM100 and
  SM107.  Past those rows the rows (``T * H``) and the SM count pick the
  kernel by bands of rows per SM.  The chunked backwards of an FP32 weight
  gradient run over the chunks of rows of the deterministic backward at
  FlashInfer 2e989556 (8-row blocks, ``max(1, ceil(blocks / 8192))`` blocks a
  chunk), at most 8 CTAs per SM: each chunk's FP32 weight-gradient partial is
  that backward's, and the partials are added exactly, as signed 64-bit
  integer windows of their FP32 values (an addition of integers does not
  depend on its order); the CTA that arrives last rounds the exact sum once
  into the weight gradient.  So that weight gradient is the same bits at any
  grid, that package's bits except where the exact sum lies within its FP64
  sum's error of a rounding boundary, and an infinite or ``NaN`` partial gives
  its column the class the FP64 sum gives it.  The static backward (CTA ``b``
  runs the chunks ``b``, ``b + grid``, ...) runs up to 256 rows per SM on
  SM100 and SM103 and up to 64 on SM107; on SM107 it keeps its weight-gradient
  windows in 4 copies plus occupancy words.  Past the static band the claimed
  backward runs, whose CTAs claim runs of chunks from the workspace's claim
  counter, so a CTA that starts late takes fewer; on SM100 and SM103 its
  thread 0 warms the CTA's next block in L2 with one bulk prefetch per tensor
  where ``dy``, ``x`` and ``z`` hold packed rows (other layouts keep
  per-thread prefetches), on SM107 it warms the block two ahead.
- Past the one-cluster backwards, a BF16 weight gradient (or
  ``compute_grad_weight=False``) runs the fused backward on every
  architecture: past 512 rows on SM100, past 256 rows on SM103, up to 1000
  rows per SM; on SM107 past 64 up to 256 rows per SM, where it makes the
  step faster than the repeatable backward.  An FP32 weight gradient runs no
  fused backward.  It
  runs the deterministic package's fused backward's schedule and row
  arithmetic: 256-thread CTAs (16 lanes a row, 16-row blocks,
  ``min(ceil(rows / 16), 4 * SMs)`` of them), which claim the blocks in chunks
  when there are more blocks than CTAs, with that backward's approximate gate
  and ``rstd``, so ``dx`` and ``dz`` are the deterministic package's bits at
  any grid (within the bound above).  It adds each thread's FP32 runs of at
  most 16 rows, folded into FP32 two-sum pairs, to FP64 accumulators with
  atomics; the CTA that arrives last rounds the sum to FP32, then to BF16.
  The order of that sum depends on scheduling, so the BF16 weight gradient's
  last bits may vary from run to run, within the family's error budget.
- Past 1000 rows per SM a BF16 weight gradient (or
  ``compute_grad_weight=False``) runs the repeatable backward on SM100 and
  SM103 (on SM107 past 512 rows up to 64 rows per SM and past 256), this
  package's copy of the deterministic package's BF16
  backward: the fused backward's grid and row arithmetic, its CTAs claiming
  the blocks in chunks whose rows are fixed by the row count and the grid,
  each chunk's weight-gradient terms added exactly into fixed-point digits in
  the workspace and the sum rounded once by the CTA that arrives last.  Its
  ``dx``, ``dz`` and weight gradient are the deterministic package's bits,
  the same for the rows and the SM count: ``dx`` and ``dz`` within the bound
  above, the weight gradient not the sum of the chunk partials of the
  deterministic backward at FlashInfer 2e989556 (within that package's
  budget).  Packed rows (``grad_out``, ``x`` and ``z`` each with head stride
  128 and token stride ``H * 128``) run its copy of the deterministic
  package's look-ahead program, the same build, from 2560 rows per SM on
  SM100 and SM103 and from 1536 on SM107, with the same bits.

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
3. Under ``torch.use_deterministic_algorithms(True)`` a backward call of this
   module that computes a BF16 weight gradient in the fused band (above)
   raises PyTorch's alert immediately before that launch (``RuntimeError``;
   with ``warn_only=True`` a ``UserWarning``, then the call runs); no other
   call of this module alerts.  The check runs on the host: a CUDA Graph
   checks it when it is captured, not when it is replayed.
4. Autograd: ``CakeRMSNormSiLUGateDetFunction`` and
   ``CakeRMSNormSiLUGateNondetFunction`` save only ``x``, ``z`` and ``weight``
   and run the backward of their own package.  A backward workspace serves
   only its own package (FP32 for the deterministic package, ``torch.uint8``
   for the non-deterministic one): each backward refuses the other package's
   workspace (``ValueError``) in every call, also with no rows.
5. The former names are gone, with no alias or shim: the former modules fail
   at import, and the former names without a mode word are attributes of
   neither ``flashinfer`` nor either module.  The SiLU-gated RMSNorm section
   of ``docs/api/norm.rst`` maps each former name to its names in both modes.

The non-deterministic backward's workspace
(``cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, device)``, a
zero-filled ``torch.uint8`` tensor of 29192 bytes on SM100 and SM103 and 36360
bytes on SM107, whatever ``T * H``: the largest layout of the architecture's
backward kernels) holds, for a chunked backward, the integer windows of the
exact sum, 128 FP32 class words, the arrival ticket and the claim counter,
and, for the fused backward, its FP64 accumulators and the same two words
(4104 bytes), and, for the repeatable backward, its fixed-point digits, a
flag word a column and the same two words (18952 bytes); every launch of the
backward uses one, also with ``compute_grad_weight=False``, but the
one-cluster backwards of a call of few
rows, which read none.  The kernel leaves it zero: allocate it once, reuse it
across calls, shapes, both weight-gradient dtypes and CUDA Graph replays
without a reset, and give each stream that may run a backward at the same
time its own buffer.  A buffer that did not come from that factory, or that
anything else wrote into, must be zeroed with
``cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)`` before its next
backward.  It is not interchangeable with the deterministic package's FP32
buffer: each backward refuses the other package's buffer (``ValueError``) in
every call, also with no rows.
Calls without a workspace use one buffer per device, stream and backward
kernel that takes one (one for each static, fused, repeatable, row-claims
and claimed backward its architecture routes; none for the one-cluster
backwards), kept for later calls; the autograd node retains nothing else.  They also keep a reserve of 64 zero-filled buffers per device (about
1.8 MiB on SM100 and SM103, 2.2 MiB on SM107, filled outside any capture): a call during CUDA Graph capture takes
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
    ``rsqrt.approx`` and the gate from ``ex2.approx`` and ``rcp.approx`` (the
    deterministic forward's split gate), every program's arithmetic fixed, so
    ``y`` repeats bit for bit for the same inputs on the same device (its bits
    need not be the deterministic package's; that one is
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_forward`).
    A forward with 16 threads to a row runs up to 38 rows per SM on SM100,
    36 on SM103 and 65 on SM107 where ``x`` and ``z`` hold packed rows (row
    ``n`` at element ``128 n``), up to 12 on SM100 / SM103 (59 on SM107) for
    other layouts, and past that, in bands of rows per SM, the pipelined
    forwards, forwards whose CTAs each walk several row blocks and (other
    layouts past 700 rows per SM on SM100 / SM103, 480 on SM107) a forward of
    one CTA per 8-row block; the programs associate a row's sum of
    squares differently (the row count, the SM count and the layout pick the
    program), so ``y``'s bits can differ between bands.  The forward does not alert under
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
    claim counter (the fused backward's FP64 accumulators and two words and
    the repeatable backward's fixed-point digits, flag words and two words fit
    within it), whatever ``rows``: about 29 KiB on SM100 and SM103 (29192
    bytes) and 36 KiB on SM107 (36360 bytes).

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
    each backward refuses the other package's buffer in every call, also with
    no rows.

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

    Zeroes the integer windows, the class words, the FP64 accumulators, the
    fixed-point digits, the flag words, the arrival ticket and the claim
    counter (a contiguous ``torch.uint8`` CUDA tensor of at least
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

    One kernel of the non-deterministic package (the rows, the SM count and
    the dtype of ``dw`` pick it; see the module): ``dx`` and ``dz`` are one
    row arithmetic in every band but the fused and the repeatable ones (there
    the deterministic package's bits), within one BF16 ulp of the FP64 value
    plus 1/64 ulp of the tensor's largest finite magnitude (in every band).  A
    chunked backward (an FP32 ``dw``) runs over the chunks of rows of the
    deterministic backward at FlashInfer 2e989556: each chunk's FP32
    weight-gradient partial is that backward's, and the partials are added
    exactly as 64-bit integer windows; the CTA that arrives last rounds the
    exact sum once into ``dw`` (adding
    ``grad_weight``'s value first when accumulating), so ``dw`` is that
    backward's bits except where the exact sum lies within its FP64 sum error
    of a rounding boundary.  A call of few rows runs one cluster of 16 CTAs
    (``dx`` and ``dz`` the same arithmetic) that adds the partials of a ``dw``
    column in a fixed order in FP64 and uses no workspace: up to 256 rows
    with a BF16 ``dw`` or ``compute_grad_weight=False`` (the
    short backward, whose ``dw`` repeats its bits, which need not be the
    deterministic backward's), else up to 512 rows (the exact tiny backward,
    the bits of the deterministic backward at FlashInfer 2e989556; SM103 runs
    it for an FP32 ``dw`` only).  The fused band (a BF16 ``dw`` or
    ``compute_grad_weight=False`` past 512 rows on SM100 and past 256 rows on
    SM103 up to 1000 rows per SM, on SM107 past 64 up to 256 rows per SM; an
    FP32 ``dw`` has none) runs
    the fused backward, the deterministic package's fused backward's schedule
    and row arithmetic (``dx`` and ``dz`` its bits at any grid) with FP32 runs
    of rows added to FP64 accumulators with atomics: that ``dw`` stays within
    the family's error budget, but its last bits may vary from run to run.
    Past 1000 rows per SM (on SM107 past 512 rows up to 64 rows per SM and past
    256) the repeatable backward, this package's copy of the deterministic
    package's backward, adds each chunk's weight-gradient terms
    exactly into fixed-point digits in the workspace: ``dx``, ``dz`` and
    ``dw`` the deterministic package's bits, the same for the rows and the SM
    count (packed rows from 2560 rows per SM on SM100 and SM103 and from 1536
    on SM107 run its look-ahead program, with the same bits).  Every other
    output repeats bit for bit, whatever the order in
    which CTAs run and, but in the repeatable band, whatever the grid.
    The deterministic backward is
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward`.
    Under ``torch.use_deterministic_algorithms(True)`` a call that computes a
    BF16 ``dw`` in the fused band raises PyTorch's alert (``RuntimeError``;
    a ``UserWarning`` under ``warn_only=True``, then the call runs) after the
    argument checks, immediately before the launch; no other call alerts.

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
        rounding (exactly on the static, the claimed and the repeatable
        backward; in FP64 before the rounding in the small-row and the fused
        bands;
        needs ``grad_weight``).
        A call with no rows leaves the accumulator as it is.
    compute_grad_weight : bool
        ``False`` skips the weight-gradient reduction and returns ``None`` for
        it.  It takes the BF16 bands: a call of up to 512 rows (256 on SM103)
        runs a one-cluster backward, which uses no workspace (a ``workspace``
        passed is checked and left as it is).
        With more rows the kernel touches neither ``grad_weight`` nor the
        workspace's windows or accumulators, but its CTAs still take the
        arrival ticket (and in the claimed, the fused and the repeatable
        backwards claim their chunks or blocks from the claim counter), so it
        uses a workspace
        too (and leaves it zero).
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

    Every output of the node repeats bit for bit but a BF16 weight gradient in
    the fused band, whose last bits may vary from run to run; under
    ``torch.use_deterministic_algorithms(True)`` the backward alerts
    immediately before computing the weight gradient in such a band, and
    nothing else alerts.  A non-reentrant
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
