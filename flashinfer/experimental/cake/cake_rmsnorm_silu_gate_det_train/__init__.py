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

Backend of :mod:`flashinfer.cake_rmsnorm_silu_gate_det_train`: the Cake SiLU-gated
RMSNorm training forward / backward (SM100 / SM103 / SM107) as one
self-contained package (host API bodies, the ``cake_jit`` registry, the pinned
``cake_build`` loader and the generated CUDA sources under ``csrc/``).  The
``@flashinfer_api``-decorated entry points and the user-facing documentation
live in :mod:`flashinfer.cake_rmsnorm_silu_gate_det_train`; import that module.

Host design (every control is a keyword; no environment variable):

- ``x``, ``z`` and the upstream gradient are BF16 tensors of one shape
  ``[..., 128]``: ``[T, H, 128]``, the flattened ``[T * H, 128]`` or any other
  leading shape.  Row ``n = t * H + h`` of a ``[T, H, 128]`` view is normalized
  on its own; every tensor takes its own token and head strides, so a gate that
  is a column slice of a wider projection is read in place.  A layout the
  kernels cannot read (a non-unit last stride, a stride or storage offset that
  is not a multiple of 16 elements) is materialized once as a contiguous copy.
- ``weight`` is BF16 ``[128]``, shared by every token and head.
- The forward saves nothing beyond its inputs; the backward recomputes the
  statistics and the gate from ``x`` and ``z``.
- Forward: 128-thread CTAs of 16-row blocks (``cake_jit.launch_grid``): one
  CTA per block below 8192 rows per SM, and from there the fewest whole waves
  of 6 CTAs per SM that give no CTA more than 2 blocks, a CTA loading its next
  block while it computes the current one.  One module per architecture, and
  on SM107 a second one for packed rows (``x`` and ``z`` both read as
  contiguous ``[T, H, 128]``; ``cake_jit.forward_module``) that computes a row
  with the same FP32 operations in the same order, so the layout changes no
  bits.  The sum of squares runs in one fixed order per architecture (on
  SM107 as four interleaved chains).  ``rstd`` comes from the approximate FP32
  reciprocal square root, and ``silu(z) = z * (1 / (1 + 2^(-z * log2(e))))`` from the
  approximate FP32 base-2 exponential and reciprocal (each a few FP32 units in
  the last place from the exact value), then one rounding to BF16.  So ``y``
  can differ from the correctly rounded FP32 formula by one BF16 unit in the
  last place where that value lies within a few FP32 units in the last place
  of a BF16 rounding boundary.  For finite ``z`` below about ``-87.3`` the
  reciprocal flushes to zero and the gate is ``-0`` (the exact gate is below
  ``2^-119`` in magnitude there).
- Backward: one kernel, BF16 or FP32 weight gradient.  It launches ``G =
  min(row blocks, 4 * SMs)`` 256-thread CTAs over blocks of 16 rows
  (``cake_jit.launch_grid``): CTA ``c`` takes block ``c`` when ``G`` covers
  every block; otherwise every CTA claims chunks of blocks (interleaved within
  each halving round) from a counter in the workspace, following a table of
  halving rounds that every CTA computes from the block count and ``G``, so a
  CTA that starts late takes fewer blocks.  From 2560 rows per SM on SM100 and
  SM103 and 1536 on SM107, a call whose ``grad_out``, ``x`` and ``z`` the
  kernel reads as packed rows (head stride 128, token stride ``H * 128``;
  ``cake_jit.backward_module``) runs the same backward in a loop that loads a
  block's rows while the previous block's row math runs: the same grid, FP32
  operations in the same order and workspace, so the same bits; other calls run
  the default module of their weight-gradient dtype (``cake_jit.ROUTES``).
  ``dx`` and ``dz`` are FP32 formulas rounded once each, with ``rstd`` from the
  approximate FP32 reciprocal square root and the sigmoid factor ``s`` from the
  approximate FP32 base-2 exponential of the exact FP32 product of ``-z`` and
  the 15-bit head of ``log2(e)`` (the remaining bits applied to first order)
  and the approximate reciprocal, on every element alike: each lies within one
  BF16 unit in the last place of the same formula in correctly rounded FP32
  plus 1/64 of a unit of its row's scale (for ``dz`` the row's largest
  ``|dy * x * rstd * w|``, for ``dx`` the sum of the largest magnitudes of its
  two terms; rows whose sums reach ``2^127`` aside).  Like the FP32 sigmoid,
  ``s`` is subnormal for ``z`` from about ``-87.34`` to ``-88.72`` and 0 below
  that (the gate ``z * s`` is then ``-0``).  For the weight gradient each thread
  sums its rows' terms ``a * n`` of a chunk in FP32 runs of at most 16 rows
  and folds each run into an FP32 two-sum pair (a sum and its exact error
  term); each CTA adds its pairs exactly into 64-bit fixed-point digit
  counters and adds those to the workspace with integer atomics, which are
  exact in any order.  The CTA that arrives last adds ``grad_weight``'s value
  exactly as digits when accumulating, rounds each column's exact sum once to
  FP32 (round to nearest even, in integer arithmetic), stores it (a BF16
  ``grad_weight`` takes that FP32 value rounded to BF16, as an FP32 sum cast
  to BF16) and zeroes the workspace.  A chunk's runs follow from the row count
  and ``G`` alone, whichever CTA takes it, so the weight gradient is the same
  bits at every call with the same row count on a device with the same SM
  count; each run errs by at most about ``2^-20`` of its terms' magnitude (the
  approximate gate and ``rstd`` put each term within a few FP32 units in the
  last place of the FP32 formula's term) and the digits sum exactly.
- Workspace: ``cake_jit.workspace_elements`` FP32 elements (18952 bytes): the
  digit counters (``2 * 9 * 128`` 64-bit words, read as ``uint64``), one
  32-bit flag word a column (infinite and ``NaN`` sums), and two 32-bit words,
  the arrival ticket and the claim counter.  Every launch uses it, also one
  with ``compute_grad_weight=False`` (its CTAs still take tickets and claim
  blocks), and leaves all of it zero, so a workspace zeroed once serves every
  later call on its stream, of any shape and either weight-gradient dtype, and
  every CUDA Graph replay, with no host write.  A caller workspace must come
  from :func:`cake_rmsnorm_silu_gate_det_train_backward_workspace`
  (zero-filled), or be zeroed with
  :func:`cake_rmsnorm_silu_gate_det_train_reset_workspace` after anything else
  wrote into it (a launch that did not complete, another use of the buffer).
- Retained state: the autograd function saves only ``x``, ``z`` and ``weight``
  through ``save_for_backward``, so a non-reentrant checkpointed forward keeps
  nothing but its output.  A backward without a caller workspace uses the
  zero-filled workspace of its (device, stream), allocated on first use and
  kept in ``_STREAM_WORKSPACES`` for later calls (18952 bytes for every stream
  that ran a backward).  Such eager calls also keep a reserve of zero-filled
  workspaces per device, filled outside any capture in allocations of 64
  (about 1.2 MiB each) whenever fewer than 32 are ready, up to 1024 per
  process (about 18.75 MiB), from which a call during CUDA Graph capture takes
  the workspace of its capture and capture stream for good, so a replay
  launches no fill kernel (see ``_RESERVES``; with none ready, the graph
  zero-fills one of its own).
- No call synchronizes the host or the device, and every kernel runs on the
  current torch stream.  No launch is cooperative or uses programmatic
  dependent launch, and no CTA waits for another CTA: beside a kernel resident
  on another stream, a backward CTA that is not resident yet starts as any of
  the backward's CTAs exits and claims only the blocks left.
- Mode: the mode is part of the name (``det``).  Every entry point is
  deterministic and takes no ``deterministic`` keyword (passing one is a
  ``TypeError``); every launch gives the same bits for the same inputs on
  the same device (the backward's sum across CTAs is exact integer addition,
  whose order does not change it), so ``torch.use_deterministic_algorithms``
  does not affect any call.  The
  backward refuses a workspace of the non-deterministic package
  (``flashinfer.cake_rmsnorm_silu_gate_nondet_train``; an FP64 or ``uint8``
  tensor) with a ``ValueError``, also in a call with no rows.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ALIGNMENT_ELEMENTS,
    ARCH_BY_CAPABILITY,
    ARCHES,
    HEAD_DIM,
    INT32_MAX,
    LONG_ROLES,
    ROLES,
    ROUTES,
    arch_for_capability,
    backward_module,
    device_sm_count,
    workspace_elements,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
_ALIGNMENT_BYTES = ALIGNMENT_ELEMENTS * 2

# The workspace factories the backward's refusal of the other mode's workspace
# names: this package's FP32 workspace and the workspace of the
# non-deterministic package (named only; nothing is imported from it).
_DETERMINISTIC_WORKSPACE = (
    "flashinfer.cake_rmsnorm_silu_gate_det_train."
    "cake_rmsnorm_silu_gate_det_train_backward_workspace"
)
_NONDETERMINISTIC_WORKSPACE = (
    "flashinfer.cake_rmsnorm_silu_gate_nondet_train."
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace"
)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _arch_of_index(device_index: int) -> Optional[str]:
    return arch_for_capability(tuple(torch.cuda.get_device_capability(device_index)))


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(
            f"Cake SiLU-gated RMSNorm training requires CUDA tensors, got {device}"
        )
    arch = _arch_of_index(_device_index(device))
    if arch is None:
        raise ValueError(
            "Cake SiLU-gated RMSNorm training supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got "
            f"{tuple(torch.cuda.get_device_capability(device))}"
        )
    return arch


def is_cake_rmsnorm_silu_gate_det_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of an exported architecture."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    return _arch_of_index(_device_index(device)) is not None


def _rows_shape(shape: torch.Size) -> tuple[int, int]:
    """``(T, H)`` of a ``[..., 128]`` shape: ``[T, H, 128]`` as given, a 2-D
    ``[N, 128]`` as ``(N, 1)``, deeper leading dimensions folded into ``T``."""

    if len(shape) < 1 or shape[-1] != HEAD_DIM:
        raise ValueError(f"expected a [..., {HEAD_DIM}] tensor, got {tuple(shape)}")
    if len(shape) == 1:
        return 1, 1
    if len(shape) == 2:
        return int(shape[0]), 1
    heads = int(shape[-2])
    tokens = 1
    for size in shape[:-2]:
        tokens *= int(size)
    return tokens, heads


def _aligned(value: int) -> bool:
    return value % ALIGNMENT_ELEMENTS == 0


def _kernel_view(
    tensor: torch.Tensor, name: str, *, shape: torch.Size, device: torch.device
) -> tuple[torch.Tensor, int, int]:
    """``(view, token stride, head stride)`` the kernels read for one input.

    The tensor is viewed as ``[T, H, 128]``; it is read in place when its last
    stride is 1, its storage offset and both strides are multiples of 16
    elements (32 bytes) and its storage is 32-byte aligned, else copied once.
    A stride of a size-1 dimension is never dereferenced: a single token's
    stride is passed as 0 and a single head's as 128, the stride of rows that
    follow one another, so a ``[T, 1, 128]`` or ``[T, 128]`` input whose token
    stride is 128 passes the persistent forwards' packed-rows test (head
    stride 128 and token stride ``H * 128``: they then warm their next row
    blocks in L2) as a ``[T, H, 128]`` input with packed rows does.  The
    strides change no bits.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
    if tensor.shape != shape:
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    tokens, heads = _rows_shape(shape)
    view = None
    try:
        view = tensor.view(tokens, heads, HEAD_DIM)
    except RuntimeError:
        view = None
    if view is not None:
        tok = int(view.stride(0)) if tokens > 1 else 0
        head = int(view.stride(1)) if heads > 1 else HEAD_DIM
        if (
            view.stride(2) == 1
            and _aligned(tok)
            and _aligned(head)
            and view.data_ptr() % _ALIGNMENT_BYTES == 0
        ):
            return view, tok, head
    copy = tensor.reshape(tokens, heads, HEAD_DIM).clone(
        memory_format=torch.contiguous_format
    )
    return (
        copy,
        heads * HEAD_DIM if tokens > 1 else 0,
        HEAD_DIM,
    )


def _weight(weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if weight.device != device:
        raise ValueError(f"weight must live on {device}, got {weight.device}")
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"weight must be torch.bfloat16, got {weight.dtype}")
    if tuple(weight.shape) != (HEAD_DIM,):
        raise ValueError(
            f"weight must have shape ({HEAD_DIM},), got {tuple(weight.shape)}"
        )
    if weight.is_contiguous() and weight.data_ptr() % _ALIGNMENT_BYTES == 0:
        return weight
    return weight.clone(memory_format=torch.contiguous_format)


def _output(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(kernel buffer, caller's tensor to copy into or None)``: the kernels
    write contiguous 32-byte aligned rows; a caller's tensor of another layout
    receives a copy after the launch."""

    if tensor is None:
        return torch.empty(shape, dtype=dtype, device=device), None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device or tensor.dtype != dtype or tensor.shape != shape:
        raise ValueError(
            f"{name} must be a {dtype} tensor of shape {tuple(shape)} on {device}, "
            f"got {tensor.dtype} {tuple(tensor.shape)} on {tensor.device}"
        )
    if tensor.is_contiguous() and tensor.data_ptr() % _ALIGNMENT_BYTES == 0:
        return tensor, None
    return torch.empty(shape, dtype=dtype, device=device), tensor


def _check_rows(tokens: int, heads: int) -> int:
    rows = tokens * heads
    if rows > INT32_MAX:
        raise ValueError(
            f"T * H = {rows} rows exceeds the kernels' 32-bit row index ({INT32_MAX})"
        )
    return rows


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
    square root, base-2 exponential and reciprocal (see the module); one
    rounding to BF16 at the end.  The same inputs give the same bits at every
    call.

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

    device = x.device
    arch = _device_arch(device)
    shape = x.shape
    tokens, heads = _rows_shape(shape)
    rows = _check_rows(tokens, heads)
    w = _weight(weight, device)
    y, y_user = _output(out, "out", shape=shape, device=device)
    if rows == 0:
        return y if y_user is None else y_user
    xv, x_tok, x_head = _kernel_view(x, "x", shape=shape, device=device)
    zv, z_tok, z_head = _kernel_view(z, "z", shape=shape, device=device)
    values = {
        "x": xv,
        "z": zv,
        "w": w,
        "y": y,
        "rows": rows,
        "heads": heads,
        "x_tok": x_tok,
        "x_head": x_head,
        "z_tok": z_tok,
        "z_head": z_head,
        "eps": float(eps),
    }
    index = _device_index(device)
    name = _loader.forward_module(
        arch, rows, heads, device_sm_count(index), x_tok, x_head, z_tok, z_head
    )
    _loader.run_module(name, values, rows=rows, device_index=index)
    if y_user is not None:
        y_user.copy_(y)
        return y_user
    return y


def _workspace_elements(arch: str) -> int:
    """FP32 elements of a workspace that serves every backward module of
    ``arch`` (both weight-gradient dtypes and their long modules, any row
    count)."""

    routes = ROUTES[arch]
    return max(
        workspace_elements(routes[role])
        for role in ROLES + LONG_ROLES
        if role.startswith("bwd") and role in routes
    )


def cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
    rows: int, device: torch.device
) -> int:
    """Bytes of the caller-owned backward workspace for ``rows`` rows
    (``T * H``) on ``device``: the weight gradient's fixed-point digit
    counters, one flag word a column, the arrival ticket and the claim counter,
    18952 bytes whatever ``rows``."""

    if int(rows) < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    return _workspace_elements(_device_arch(torch.device(device))) * 4


def cake_rmsnorm_silu_gate_det_train_backward_workspace(
    rows: int, device: torch.device
) -> torch.Tensor:
    """A caller-owned backward workspace for ``rows`` rows (``T * H``): a
    zero-filled FP32 tensor of
    :func:`cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes` bytes.

    The backward leaves it zero after every call, so it serves every later call
    on its stream (any ``T`` and ``H``, BF16 or FP32 weight gradient) and every
    CUDA Graph replay without a reset.  A backward that may run at the same time
    on another stream needs its own workspace.  The non-deterministic package's
    workspace is not a workspace of this package (and this one not of that
    package's backward).
    """

    device = torch.device(device)
    if int(rows) < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    return torch.zeros(
        _workspace_elements(_device_arch(device)), dtype=torch.float32, device=device
    )


def cake_rmsnorm_silu_gate_det_train_reset_workspace(
    workspace: torch.Tensor,
) -> None:
    """Return a caller-owned backward workspace to its first-use state: zero all
    of it (the digit counters, the flag words, the arrival ticket and the claim
    counter).

    Every backward leaves its workspace zero, so reuse across calls, shapes,
    weight-gradient dtypes and CUDA Graph replays needs no reset.  A workspace
    that did not come from
    :func:`cake_rmsnorm_silu_gate_det_train_backward_workspace`, or that
    something else wrote into (a launch that did not complete, another use of
    the buffer), must be reset before its next backward.  The fill runs on the
    current stream of the workspace's device, without a host synchronization
    (it can be captured in a CUDA Graph).  The workspace must be a contiguous,
    16-byte aligned FP32 CUDA tensor of at least
    :func:`cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes` bytes; a
    workspace of the non-deterministic package is refused with a
    ``ValueError``.
    """

    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace must be a torch.Tensor")
    device = workspace.device
    _check_workspace(workspace, device, _workspace_elements(_device_arch(device)))
    workspace.zero_()


# The workspaces of calls without a caller workspace: one per (device, stream),
# zero-filled once and kept for the process (18952 bytes for every stream that
# ran a backward).  Every backward leaves its workspace zero and the calls of
# one stream run in order, so they share it (no allocation and no fill per
# call); every backward module (both weight-gradient dtypes and their long
# modules) reads the same layout.  A call during CUDA Graph capture uses a
# workspace of its capture instead (``_capture_workspace``), so a graph
# replayed on another stream never shares a buffer with eager calls or other
# graphs.
_STREAM_WORKSPACES: dict[
    tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}

# The workspaces of captured calls without a caller workspace.  A replay must
# find its workspace zero when the backward kernel starts and no other launch
# may use it, yet a fill inside the graph adds a kernel to every replay.  So
# eager calls without a caller workspace keep a reserve of zero-filled
# workspaces per device, filled outside any capture: ``_RESERVE_SIZE`` at a
# time from one allocation, each ready once a later eager call has seen its
# fill complete (``Event.query``).  The first such call of a capture on a
# capture stream takes a ready one for good and adds no node to the graph: the
# first replay finds it zero, every launch leaves it zero, launches of one
# graph never overlap, and no other launch gets it.  Later calls of the same
# capture on the same capture stream run after it in the graph and reuse it
# (capture ids from cuda-python; without it, each captured call takes its
# own).  No graph can tell this package when it is destroyed, so the reserve's
# memory is kept for the process: at most ``_GRAPH_WORKSPACE_LIMIT``
# workspaces per device.  With none ready (the reserve still filling or used
# up), a captured call zero-fills a workspace from the graph's pool inside the
# graph (one fill kernel per call and replay).  ``_RESERVE_LOCK`` serializes
# the reserve's bookkeeping between threads, so a workspace moves to ready
# once, after its fill was seen complete.
_RESERVE_SIZE = 64
_GRAPH_WORKSPACE_LIMIT = 1024
# device index -> {"ready": [workspace, ...], "pending": [(event recorded after
# the fill, [workspace, ...]), ...], "blocks": [allocation, ...], "carved": int}
_RESERVES: dict[int, dict[str, Any]] = {}
# (device index, capture stream) -> (capture id, the workspace of that capture)
_CAPTURE_WORKSPACES: dict[tuple[int, int], tuple[int, torch.Tensor]] = {}
_RESERVE_LOCK = threading.Lock()


def _refill_reserve(arch: str, device: torch.device) -> None:
    """Eager calls: move filled reserve workspaces to ready and, below half the
    reserve and under the limit, zero-fill ``_RESERVE_SIZE`` more on the
    current stream (no host synchronization)."""

    index = _device_index(device)
    with _RESERVE_LOCK:
        reserve = _RESERVES.get(index)
        if reserve is None:
            reserve = _RESERVES[index] = {
                "ready": [],
                "pending": [],
                "blocks": [],
                "carved": 0,
            }
        pending = reserve["pending"]
        while pending and pending[0][0].query():
            reserve["ready"].extend(pending.pop(0)[1])
        waiting = sum(len(workspaces) for _, workspaces in pending)
        count = min(_RESERVE_SIZE, _GRAPH_WORKSPACE_LIMIT - reserve["carved"])
        if 2 * (len(reserve["ready"]) + waiting) >= _RESERVE_SIZE or count <= 0:
            return
        elements = _workspace_elements(arch)
        stride = -(-elements // 64) * 64  # 256-byte aligned workspaces
        block = torch.zeros(count * stride, dtype=torch.float32, device=device)
        filled = torch.cuda.Event()
        filled.record(torch.cuda.current_stream(device))
        reserve["blocks"].append(block)
        reserve["carved"] += count
        pending.append(
            (filled, [block[i * stride : i * stride + elements] for i in range(count)])
        )


@functools.cache
def _cudart() -> Any:
    """cuda-python's runtime module, or None when it is not installed."""

    try:
        from cuda.bindings import runtime
    except ImportError:
        return None
    return runtime


def _capture_id(cudart: Any, stream: Any) -> Optional[int]:
    """The id of the capture in progress on ``stream``, or None."""

    result = cudart.cudaStreamGetCaptureInfo(cudart.cudaStream_t(stream.cuda_stream))
    if (
        result[0] != cudart.cudaError_t.cudaSuccess
        or result[1] != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    ):
        return None
    return int(result[2])


def _capture_workspace(arch: str, device: torch.device) -> torch.Tensor:
    """The FP32 workspace of a call without a caller workspace during CUDA
    Graph capture on the current stream of ``device``."""

    index = _device_index(device)
    stream = torch.cuda.current_stream(device)
    cudart = _cudart()
    capture = None if cudart is None else _capture_id(cudart, stream)
    key = (index, stream.cuda_stream)
    if capture is not None:
        held = _CAPTURE_WORKSPACES.get(key)
        if held is not None and held[0] == capture:
            return held[1]
    with _RESERVE_LOCK:
        reserve = _RESERVES.get(index)
        workspace = reserve["ready"].pop() if reserve and reserve["ready"] else None
    if workspace is None:
        # Inside the graph: from its pool, zero-filled at every replay.
        return torch.zeros(
            _workspace_elements(arch), dtype=torch.float32, device=device
        )
    if capture is not None:
        _CAPTURE_WORKSPACES[key] = (capture, workspace)
    return workspace


def _stream_workspace_views(
    name: str, arch: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(digit counters, flag words, ticket and claim words)`` of backward
    module ``name`` for a call without a caller workspace."""

    if torch.cuda.is_current_stream_capturing():
        return _workspace_views(_capture_workspace(arch, device), name, device)
    _refill_reserve(arch, device)
    key = (_device_index(device), torch.cuda.current_stream(device).cuda_stream)
    views = _STREAM_WORKSPACES.get(key)
    if views is None:
        views = _workspace_views(
            torch.zeros(_workspace_elements(arch), dtype=torch.float32, device=device),
            name,
            device,
        )
        _STREAM_WORKSPACES[key] = views
    return views


def _refuse_nondeterministic_workspace(workspace: Optional[torch.Tensor]) -> None:
    """Refuse a workspace of the non-deterministic package (an FP64 or uint8
    tensor), naming both modes' workspaces."""

    names = {torch.float64: "FP64", torch.uint8: "uint8"}
    if isinstance(workspace, torch.Tensor) and workspace.dtype in names:
        raise ValueError(
            f"workspace is {names[workspace.dtype]}, a workspace of the "
            "non-deterministic "
            f"mode ({_NONDETERMINISTIC_WORKSPACE}); the deterministic backward "
            "needs its own zero-filled FP32 workspace "
            f"({_DETERMINISTIC_WORKSPACE})"
        )


def _check_workspace(
    workspace: torch.Tensor, device: torch.device, elements: int
) -> torch.Tensor:
    """The flat view of a caller's workspace of at least ``elements`` FP32
    elements on ``device``; a workspace of the non-deterministic package is
    refused naming both modes' workspaces."""

    _refuse_nondeterministic_workspace(workspace)
    if (
        not isinstance(workspace, torch.Tensor)
        or workspace.device != device
        or workspace.dtype != torch.float32
        or not workspace.is_contiguous()
        or workspace.numel() < elements
        or workspace.data_ptr() % 16 != 0
    ):
        raise ValueError(
            f"workspace must be a contiguous, 16-byte aligned FP32 tensor on {device} "
            f"with at least {elements} elements, zero before its first use "
            "(cake_rmsnorm_silu_gate_det_train_backward_workspace(rows, device), "
            "or cake_rmsnorm_silu_gate_det_train_reset_workspace(workspace))"
        )
    return workspace.view(-1)


def _workspace_views(
    workspace: torch.Tensor, name: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(acc, flags, done)`` of backward module ``name`` on a workspace: the
    ``uint64`` digit counters, the ``uint32`` flag word of each column, and the
    two ``uint32`` words ``[0]`` the arrival ticket and ``[1]`` the claim
    counter."""

    elements = workspace_elements(name)
    flat = _check_workspace(workspace, device, elements)
    digits = elements - HEAD_DIM - 2
    acc = flat[:digits].view(torch.uint64)
    flags = flat[digits : digits + HEAD_DIM].view(torch.uint32)
    done = flat[digits + HEAD_DIM : elements].view(torch.uint32)
    return acc, flags, done


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
    the approximate reciprocal square root, base-2 exponential and reciprocal
    (see the module): ``dx`` and ``dz`` lie within one BF16 unit in the last
    place of these formulas in correctly rounded FP32 plus 1/64 of a unit of
    the row's largest term, the same bits at every call.  ``dw`` is summed in
    FP32 runs of at most 16 rows, added exactly in fixed point in an
    order-independent way, and rounded once to FP32 (then to BF16 for a BF16
    gradient; see the module): the same row count on a device with the same SM
    count gives the same bits.

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

    device = x.device
    arch = _device_arch(device)
    shape = x.shape
    tokens, heads = _rows_shape(shape)
    rows = _check_rows(tokens, heads)
    w = _weight(weight, device)
    if accumulate_grad_weight and grad_weight is None:
        raise ValueError("accumulate_grad_weight=True needs grad_weight")
    dw = None
    if compute_grad_weight:
        if grad_weight is None:
            dw = torch.empty((HEAD_DIM,), dtype=torch.bfloat16, device=device)
        else:
            if (
                not isinstance(grad_weight, torch.Tensor)
                or grad_weight.device != device
                or grad_weight.dtype not in (torch.bfloat16, torch.float32)
                or tuple(grad_weight.shape) != (HEAD_DIM,)
                or not grad_weight.is_contiguous()
            ):
                raise ValueError(
                    f"grad_weight must be a contiguous BF16 or FP32 ({HEAD_DIM},) "
                    f"tensor on {device}"
                )
            dw = grad_weight
    dx, dx_user = _output(grad_x, "grad_x", shape=shape, device=device)
    dz, dz_user = _output(grad_z, "grad_z", shape=shape, device=device)
    # A caller's workspace is checked for any row count (its size does not
    # depend on the rows), so before the return of a call with no rows.
    if workspace is not None:
        _check_workspace(workspace, device, _workspace_elements(arch))
    if rows == 0:
        if dw is not None and not accumulate_grad_weight:
            dw.zero_()
        return (
            dx if dx_user is None else dx_user,
            dz if dz_user is None else dz_user,
            dw,
        )
    gv, dy_tok, dy_head = _kernel_view(grad_out, "grad_out", shape=shape, device=device)
    xv, x_tok, x_head = _kernel_view(x, "x", shape=shape, device=device)
    zv, z_tok, z_head = _kernel_view(z, "z", shape=shape, device=device)
    index = _device_index(device)
    role = "bwd_f32" if dw is not None and dw.dtype == torch.float32 else "bwd_bf16"
    name = backward_module(
        arch,
        role,
        rows,
        heads,
        device_sm_count(index),
        dy_tok,
        dy_head,
        x_tok,
        x_head,
        z_tok,
        z_head,
    )
    if workspace is None:
        acc, flags, done = _stream_workspace_views(name, arch, device)
    else:
        acc, flags, done = _workspace_views(workspace, name, device)
    values = {
        "dy": gv,
        "x": xv,
        "z": zv,
        "w": w,
        "dx": dx,
        "dz": dz,
        "acc": acc,
        "flags": flags,
        "done": done,
        "dw": dw if dw is not None else torch.empty((0,), dtype=w.dtype, device=device),
        "rows": rows,
        "heads": heads,
        "dy_tok": dy_tok,
        "dy_head": dy_head,
        "x_tok": x_tok,
        "x_head": x_head,
        "z_tok": z_tok,
        "z_head": z_head,
        "eps": float(eps),
        "accumulate": 1 if accumulate_grad_weight else 0,
        "weight_grad": 1 if dw is not None else 0,
    }
    _loader.run_module(name, values, rows=rows, device_index=index)
    for kernel_buffer, user_buffer in ((dx, dx_user), (dz, dz_user)):
        if user_buffer is not None:
            user_buffer.copy_(kernel_buffer)
    return (
        dx if dx_user is None else dx_user,
        dz if dz_user is None else dz_user,
        dw,
    )


class CakeRMSNormSiLUGateDetFunction(torch.autograd.Function):
    """Autograd of :func:`cake_rmsnorm_silu_gate_det`: saves ``x``, ``z`` and
    ``weight`` (no FP32 intermediate) and recomputes the rest in the backward,
    so ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only the inputs
    as saved state."""

    @staticmethod
    def forward(ctx, x, z, weight, eps):
        y = cake_rmsnorm_silu_gate_det_train_forward(x, z, weight, eps)
        ctx.save_for_backward(x, z, weight)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, z, weight = ctx.saved_tensors
        dx, dz, dw = cake_rmsnorm_silu_gate_det_train_backward(
            grad_y,
            x,
            z,
            weight,
            ctx.eps,
            compute_grad_weight=ctx.needs_input_grad[2],
        )
        return (
            dx if ctx.needs_input_grad[0] else None,
            dz if ctx.needs_input_grad[1] else None,
            dw,
            None,
        )


def cake_rmsnorm_silu_gate_det(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Differentiable ``RMSNorm(x) * weight * SiLU(z)`` over the last dimension
    (128): :class:`CakeRMSNormSiLUGateDetFunction` applied to the arguments."""

    return CakeRMSNormSiLUGateDetFunction.apply(x, z, weight, eps)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormSiLUGateDetFunction",
    "cake_rmsnorm_silu_gate_det",
    "cake_rmsnorm_silu_gate_det_train_backward",
    "cake_rmsnorm_silu_gate_det_train_backward_workspace",
    "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_det_train_forward",
    "cake_rmsnorm_silu_gate_det_train_reset_workspace",
    "is_cake_rmsnorm_silu_gate_det_train_supported",
]
