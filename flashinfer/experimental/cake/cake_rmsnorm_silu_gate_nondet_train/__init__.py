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

Backend of :mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train`: the
Cake SiLU-gated RMSNorm training forward / backward (SM100 / SM103 / SM107) of
the non-deterministic family, as one self-contained package (host API bodies,
the ``cake_jit`` registry, the pinned ``cake_build`` loader and the generated
CUDA sources under ``csrc/``).  It imports nothing from other flashinfer
packages and ships its own copy of every program it launches.  The
``@flashinfer_api`` decorations and the user-facing documentation live in
:mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train`; import that module.  The
deterministic package is ``flashinfer.cake_rmsnorm_silu_gate_det_train``: no
call of this package runs it.

Numerical contract.  Every output repeats bit for bit: the same inputs on the
same device give the same ``y``, ``dx``, ``dz`` and weight gradient on every
call, whatever the grid, the order in which CTAs run and the work beside the
call on other streams.  The forward computes ``rstd`` with ``rsqrt.approx``
and the SiLU gate with ``ex2.approx`` and ``rcp.approx`` (no flush of a
nonzero sigmoid of a normal ``z``), rounding ``y`` to BF16 once; its bits are
not the deterministic package's, and its error against an FP64 reference
stays within the family's budget.  Each forward program's arithmetic is fixed,
but the programs may associate a row's sum of squares differently (the row
count and the SM count pick the program), so ``y`` repeats bit for bit for the
same inputs on the same device.  The backward writes ``dx`` and ``dz``
with the deterministic backward's arithmetic and computes the FP32 partial of
each chunk of rows as the deterministic backward does; it adds the partials
exactly, as signed 64-bit integer windows of the FP32 values (an addition of
integers does not depend on its order), and rounds the exact sum once into
``dw`` (FP32, then BF16 for a BF16 ``dw``), so ``dw`` is the deterministic
package's bits except where the exact sum lies within its FP64 sum's error of
a rounding boundary.  An infinite or ``NaN`` partial makes its column's
``dw`` infinite or ``NaN`` with the class the FP64 sum gives it (a ``NaN``'s
sign and payload need not be its).  Up to 256 rows the backward of a BF16
weight gradient, and of a call with ``compute_grad_weight=False``, is one
thread-block cluster of 16 CTAs, its ``dx`` and ``dz`` the same arithmetic,
that sums the weight gradient on chip in a fixed order (CTA ``r`` adds every CTA's FP32
partial of its eight columns in rank order in FP64 and rounds once), the same
bits on every call.

Mode: the mode is part of the name (``nondet``).  No entry point takes a
``deterministic`` keyword (passing one is a ``TypeError``), and no call of this
package runs the deterministic package.  Every output repeats bit for bit, so
no call raises PyTorch's alert for an operation without a deterministic
implementation under ``torch.use_deterministic_algorithms(True)``.

Host design (the input layout rules are the deterministic package's):

- ``x``, ``z`` and the upstream gradient are BF16 tensors of one shape
  ``[..., 128]`` (``[T, H, 128]``, the flattened ``[T * H, 128]`` or any other
  leading shape), each with its own token and head strides; a layout the
  kernels cannot read is materialized once as a contiguous copy.
- The backward's workspace is a zero-filled ``torch.uint8`` buffer of
  :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes
  (29192 for every shipped backward that takes one) holding three buffers
  (``cake_jit.workspace_layout``): the 64-bit integer windows of the exact
  weight-gradient sum, 128 FP32 class words and the two 32-bit words of the
  arrival ticket and the claim counter.  Every launch of the static and the
  claimed backward takes a ticket (and the claimed one claims its chunks from
  the counter), also one with ``compute_grad_weight=False``, so it uses a
  workspace.  The short backward takes none (up to 256 rows with a BF16
  weight gradient or with ``compute_grad_weight=False``): it reads no
  workspace, and a
  caller workspace passed to it is checked as for any backward and left as it
  is.  The kernel leaves all of it zero, so a workspace zeroed once serves
  every later call on its stream, of any shape and either weight-gradient
  dtype, and every CUDA Graph replay, with no host write.  A caller workspace
  must come from :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`
  (zero-filled), or be zeroed with
  :func:`cake_rmsnorm_silu_gate_nondet_train_reset_workspace` after anything
  else wrote into it (a launch that did not complete, another use of the
  buffer); the kernel never needs a reset between its own calls.
- Retained state (FlashInfer issue #7039): the autograd function saves only
  ``x``, ``z`` and ``weight`` through ``save_for_backward``, so a non-reentrant
  checkpointed forward keeps nothing but its output.  A backward without a
  caller workspace that runs a module taking one (every module but the short
  backward; ``compute_grad_weight=False`` included: it runs the BF16
  weight gradient's modules, the short backward up to 256 rows) uses the
  zero-filled workspace of its (device,
  stream, backward module), allocated on first use and kept in
  ``_STREAM_WORKSPACES`` for later calls: 29192 bytes each, at most one per
  backward module (two per weight-gradient dtype: the static and the claimed
  backward) for every stream that ran such a backward.  Such eager calls also
  keep a reserve of 64 zero-filled workspaces per device (one allocation of
  about 1.8 MiB, filled outside any capture), from which a call during CUDA
  Graph capture takes the workspace of its capture and capture stream for
  good, so a replay launches no fill kernel (see ``_RESERVES``; with none
  ready, the graph zero-fills one of its own).
- Grids (``cake_jit.module_for_rows`` picks the module from the rows and
  the device's SM count): the forward launches one 64-thread CTA per block of
  8 rows; up to 12 rows per SM on sm_100a and sm_103a (59 on sm_107a) the
  short forward, one 256-thread CTA per block of 16 rows; and from more rows
  a persistent forward whose CTA ``b`` walks the blocks ``b``, ``b + grid``,
  ..., loading each next block while it computes the current one, in the
  fewest whole waves that give no CTA more than a few blocks: from 976 rows
  per SM on sm_100a and sm_103a (800 on sm_107a) the mid forward (6 CTAs per
  SM, at most 2 blocks of 16 rows each, thread 0 warming the CTA's next block
  and the first block of the CTA one wave later in L2 where x and z hold their
  rows packed: head stride 128 and token stride ``H * 128``, which a ``[T, 1,
  128]`` or ``[T, 128]`` input whose rows follow one another meets, see
  ``_kernel_view``), from 2400 (sm_100a and sm_103a) the long forward (3 CTAs
  per SM, at most 6 blocks of 32 rows each), and from 3000 on sm_103a and
  1740 on sm_107a the extra-long forward (the long forward whose thread 0
  also warms the CTA's next block in L2).  The static and the claimed
  backward run 128-thread CTAs of 8-row blocks in the deterministic
  backward's chunks (``cake_jit.backward_chunks``) and launch
  ``cake_jit.backward_ctas`` ``G = min(chunks, ctas_per_sm * SMs)`` CTAs (8
  CTAs per SM): the static backward, CTA ``b`` running the chunks ``b``,
  ``b + G``, ..., while every chunk has a CTA, and the claimed backward past
  that, thread 0 of every CTA claiming runs of chunks with a relaxed atomic
  increment on the workspace's claim counter, so a CTA that starts late takes
  fewer chunks; the BF16 short backward one cluster of 16 256-thread CTAs
  (``cake_jit.backward_ctas`` 16), CTA ``r`` taking row block ``r``, its CTAs
  scheduled together on the SMs of one GPC.  No kernel is cooperative; no
  call synchronizes the host or the device, and every kernel runs on the
  current torch stream.
- Beside kernels resident on other streams (FlashInfer issue #7548): a kernel
  holding every SM with a small shared-memory carveout delays the start of a
  kernel whose CTAs need more shared memory than is left, until it exits;
  :mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train` documents the remedy.
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
    ROLES,
    ROUTES,
    arch_for_capability,
    backward_chunks,
    backward_ctas,
    device_sm_count,
    module_for_rows,
    workspace_bytes,
    workspace_layout,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
_ALIGNMENT_BYTES = ALIGNMENT_ELEMENTS * 2


# The deterministic package, named when a caller passes its workspace (this
# package imports nothing from the deterministic package).
_DETERMINISTIC_PACKAGE = "flashinfer.cake_rmsnorm_silu_gate_det_train"
# The torch dtype of each workspace buffer (``cake_jit.WORKSPACE_BUFFERS``).
_WORKSPACE_DTYPES = {
    "dw_win": torch.uint64,
    "dw_spec": torch.float32,
    "dw_done": torch.uint32,
}
# A caller workspace starts at a multiple of this many bytes, so each buffer's
# view is aligned for its 64-bit atomics.
_WORKSPACE_ALIGNMENT_BYTES = 16


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


def is_cake_rmsnorm_silu_gate_nondet_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device with every route of this package.

    A capability probe that launches nothing; it does not depend on
    ``torch.use_deterministic_algorithms``.
    """

    device = torch.device(device)
    if device.type != "cuda":
        return False
    arch = _arch_of_index(_device_index(device))
    return arch is not None and set(ROLES) <= set(ROUTES.get(arch, {}))


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
    stride is 128 passes the kernels' packed-rows test (head stride 128 and
    token stride ``H * 128``) as a ``[T, H, 128]`` input with packed rows does.
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


def _forward(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """The forward of this package's program."""

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
    name = module_for_rows(arch, "fwd", rows, device_sm_count(index))
    _loader.run_module(name, values, rows=rows, device_index=index)
    if y_user is not None:
        y_user.copy_(y)
        return y_user
    return y


def cake_rmsnorm_silu_gate_nondet_train_forward(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""SiLU-gated RMSNorm forward of this package:
    ``y = BF16(((x * rstd) * weight) * silu(z))``.

    ``rstd`` from ``rsqrt.approx`` and the gate from ``ex2.approx`` and
    ``rcp.approx``, ``y`` rounded to BF16 once: the same bits on every call
    (not the deterministic package's bits) for the same inputs on the same
    device.  ``cake_jit.module_for_rows`` picks the program from the rows and
    the SM count (``cake_jit.ROUTES``); each program's arithmetic is fixed, but
    the programs may associate a row's sum of squares differently.  The
    arguments are those of
    :func:`flashinfer.cake_rmsnorm_silu_gate_nondet_train.cake_rmsnorm_silu_gate_nondet_train_forward`.
    """

    return _forward(x, z, weight, eps, out=out)


def _workspace_bytes(arch: str) -> int:
    """Bytes of a workspace that serves every backward module of ``arch``
    (both weight-gradient dtypes, any row count)."""

    return max(
        workspace_bytes(name)
        for role, name in ROUTES[arch].items()
        if role.startswith("bwd")
    )


def cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(
    rows: int, device: torch.device
) -> int:
    """Bytes of the caller-owned backward workspace for ``rows`` rows
    (``T * H``) on ``device``: the integer windows of the exact weight-gradient
    sum, the FP32 class words and the arrival ticket and claim counter, about
    29 KiB whatever ``rows``.
    """

    if int(rows) < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    return _workspace_bytes(_device_arch(torch.device(device)))


def cake_rmsnorm_silu_gate_nondet_train_backward_workspace(
    rows: int, device: torch.device
) -> torch.Tensor:
    """A caller-owned backward workspace for ``rows`` rows (``T * H``): a
    zero-filled ``torch.uint8`` tensor of
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes.

    The backward leaves it zero after every call, so it serves every later call
    on its stream (any ``T`` and ``H``, BF16 or FP32 weight gradient) and every
    CUDA Graph replay without a reset.  A backward that may run at the same time
    on another stream needs its own workspace.  The deterministic package's
    FP32 buffer is not a workspace of this package (and this one not of that
    package's backward).
    """

    device = torch.device(device)
    if int(rows) < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    return torch.zeros(
        _workspace_bytes(_device_arch(device)), dtype=torch.uint8, device=device
    )


def cake_rmsnorm_silu_gate_nondet_train_reset_workspace(
    workspace: torch.Tensor,
) -> None:
    """Return a caller-owned backward workspace to its first-use state: zero
    all of it (the integer windows, the class words, the arrival ticket and the
    claim counter).

    Every backward leaves its workspace zero, so reuse across calls, shapes,
    weight-gradient dtypes and CUDA Graph replays needs no reset.  A workspace
    that did not come from :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`,
    or that something else wrote into (a launch that did not complete, another
    use of the buffer), must be reset before its next backward.  The fill runs
    on the current stream of the workspace's device, without a host
    synchronization (it can be captured in a CUDA Graph).  The workspace must be
    a contiguous ``torch.uint8`` CUDA tensor of at least
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes
    starting at a 16-byte aligned address; all of it is zeroed (the
    deterministic package's FP32 buffer is refused with a ``ValueError``).
    """

    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace must be a torch.Tensor")
    device = workspace.device
    _check_workspace(workspace, device, _workspace_bytes(_device_arch(device)))
    workspace.zero_()


# The workspaces of calls without a caller workspace (every launch of the
# static and the claimed backward needs one: it takes the arrival ticket, and
# the claimed one claims its chunks; the short backward needs none): one per
# (device, stream, backward module), zero-filled once and kept for the process
# (29192 bytes each, one per backward module a stream ran).  Every backward
# leaves its workspace zero and the calls of one stream run in order, so they
# share it (no allocation and no fill per call).  A call during CUDA Graph
# capture uses a workspace of its capture instead (``_capture_workspace``), so
# a graph replayed on another stream never shares a buffer with eager calls or
# other graphs.
_STREAM_WORKSPACES: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}

# The workspaces of captured calls without a caller workspace.  A replay must
# find its workspace zero when the backward kernel starts and no other launch
# may use it, yet a fill inside the graph adds a kernel to every replay (and an
# event wait between two kernels costs more).  So eager calls without a caller
# workspace keep a reserve of zero-filled workspaces per device, filled outside
# any capture: ``_RESERVE_SIZE`` at a time from one allocation, each ready once
# a later eager call has seen its fill complete (``Event.query``).  The first
# such call of a capture on a capture stream takes a ready one for good and
# adds no node to the graph: the first replay finds it zero, every launch
# leaves it zero, launches of one graph never overlap, and no other launch gets
# it.  Later calls of the same capture on the same capture stream run after it
# in the graph and reuse it (capture ids from cuda-python; without it, each
# captured call takes its own).  No graph can tell this package when it is
# destroyed, so the reserve's memory is kept for the process: at most
# ``_GRAPH_WORKSPACE_LIMIT`` workspaces per device.  With none ready (the
# reserve still filling or used up), a captured call zero-fills a workspace from
# the graph's pool inside the graph (one fill kernel per call and replay).
# ``_RESERVE_LOCK`` serializes the reserve's bookkeeping between threads, so a
# workspace moves to ready once, after its fill was seen complete.
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
        nbytes = _workspace_bytes(arch)
        stride = -(-nbytes // 256) * 256  # 256-byte aligned workspaces
        block = torch.zeros(count * stride, dtype=torch.uint8, device=device)
        filled = torch.cuda.Event()
        filled.record(torch.cuda.current_stream(device))
        reserve["blocks"].append(block)
        reserve["carved"] += count
        pending.append(
            (filled, [block[i * stride : i * stride + nbytes] for i in range(count)])
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
    """The byte workspace of a call without a caller workspace during CUDA
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
        return torch.zeros(_workspace_bytes(arch), dtype=torch.uint8, device=device)
    if capture is not None:
        _CAPTURE_WORKSPACES[key] = (capture, workspace)
    return workspace


def _stream_workspace_views(
    name: str, arch: str, device: torch.device
) -> dict[str, torch.Tensor]:
    """The workspace buffers (``dw_win``, ``dw_spec``, ``dw_done``) of backward
    module ``name`` for a call without a caller workspace."""

    if torch.cuda.is_current_stream_capturing():
        return _workspace_views(_capture_workspace(arch, device), name, device)
    _refill_reserve(arch, device)
    key = (_device_index(device), torch.cuda.current_stream(device).cuda_stream, name)
    views = _STREAM_WORKSPACES.get(key)
    if views is None:
        views = _workspace_views(
            torch.zeros(_workspace_bytes(arch), dtype=torch.uint8, device=device),
            name,
            device,
        )
        _STREAM_WORKSPACES[key] = views
    return views


def _refuse_deterministic_workspace(workspace: Optional[torch.Tensor]) -> None:
    """Refuse an FP32 tensor (the deterministic package's workspace), naming
    both modes' workspaces."""

    if isinstance(workspace, torch.Tensor) and workspace.dtype == torch.float32:
        raise ValueError(
            "workspace is FP32, a workspace of the deterministic mode "
            f"({_DETERMINISTIC_PACKAGE}.cake_rmsnorm_silu_gate_det_train_backward_workspace); "
            "the non-deterministic backward needs its own zero-filled uint8 "
            f"workspace ({__name__}.cake_rmsnorm_silu_gate_nondet_train_backward_workspace)"
        )


def _check_workspace(
    workspace: torch.Tensor, device: torch.device, nbytes: int
) -> torch.Tensor:
    """The flat view of a caller's workspace of at least ``nbytes`` bytes on
    ``device``; an FP32 tensor (the deterministic package's workspace) is
    refused naming both modes' workspaces."""

    _refuse_deterministic_workspace(workspace)
    if (
        not isinstance(workspace, torch.Tensor)
        or workspace.device != device
        or workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
        or workspace.numel() < nbytes
        or workspace.data_ptr() % _WORKSPACE_ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"workspace must be a contiguous torch.uint8 tensor on {device} with at "
            f"least {nbytes} bytes starting at a {_WORKSPACE_ALIGNMENT_BYTES}-byte "
            "aligned address, zero before its first use "
            "(cake_rmsnorm_silu_gate_nondet_train_backward_workspace(rows, device), "
            "or cake_rmsnorm_silu_gate_nondet_train_reset_workspace(workspace))"
        )
    return workspace.view(-1)


def _workspace_views(
    workspace: torch.Tensor, name: str, device: torch.device
) -> dict[str, torch.Tensor]:
    """The buffers of backward module ``name`` on a workspace
    (``cake_jit.workspace_layout``): ``dw_win`` (uint64), ``dw_spec`` (FP32)
    and ``dw_done`` (uint32: ``[0]`` the arrival ticket, ``[1]`` the claim
    counter), each a view of its bytes."""

    flat = _check_workspace(workspace, device, workspace_bytes(name))
    views = {}
    for buffer, offset, elements in workspace_layout(name):
        dtype = _WORKSPACE_DTYPES[buffer]
        size = elements * dtype.itemsize
        views[buffer] = flat[offset : offset + size].view(dtype)
    return views


def _backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    grad_x: Optional[torch.Tensor],
    grad_z: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
    accumulate_grad_weight: bool,
    compute_grad_weight: bool,
    workspace: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The backward of this package's program."""

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
    # The other mode's workspace is refused for any row count, so before the
    # return of a call with no rows; the size checks below depend on the
    # backward module that the row count selects.
    _refuse_deterministic_workspace(workspace)
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
    sm_count = device_sm_count(_device_index(device))
    role = "bwd_f32" if dw is not None and dw.dtype == torch.float32 else "bwd_bf16"
    name = module_for_rows(arch, role, rows, sm_count)
    ctas = backward_ctas(name, rows, sm_count)
    buffers: dict[str, Optional[torch.Tensor]] = {
        "dw_win": None,
        "dw_spec": None,
        "dw_done": None,
    }
    blocks_per_chunk = 0
    if not workspace_bytes(name):
        # The short backward sums the weight gradient on chip: no workspace (a
        # caller's is checked and left as it is).
        if workspace is not None:
            _check_workspace(workspace, device, _workspace_bytes(arch))
    else:
        # Every launch of the others, weight_grad=0 included, takes the arrival
        # ticket (and the claimed backward claims its chunks from the claim
        # counter), and leaves the whole workspace zero.
        blocks_per_chunk = backward_chunks(name, rows)[0]
        if workspace is not None:
            buffers.update(_workspace_views(workspace, name, device))
        else:
            buffers.update(_stream_workspace_views(name, arch, device))
    values = {
        "dy": gv,
        "x": xv,
        "z": zv,
        "w": w,
        "dx": dx,
        "dz": dz,
        **buffers,
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
        "ctas": ctas,
        "accumulate": 1 if accumulate_grad_weight else 0,
        "weight_grad": 1 if dw is not None else 0,
        "blocks_per_chunk": blocks_per_chunk,
    }
    _loader.run_module(name, values, rows=rows, device_index=_device_index(device))
    for kernel_buffer, user_buffer in ((dx, dx_user), (dz, dz_user)):
        if user_buffer is not None:
            user_buffer.copy_(kernel_buffer)
    return (
        dx if dx_user is None else dx_user,
        dz if dz_user is None else dz_user,
        dw,
    )


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
    r"""SiLU-gated RMSNorm backward of this package: ``dx``, ``dz`` and the
    weight gradient.

    One kernel of at most ``ctas_per_sm`` CTAs per SM over the deterministic
    backward's chunks of rows (one chunk a CTA while every chunk has a CTA;
    past that the CTAs claim runs of chunks from the workspace's claim
    counter): ``dx`` and ``dz`` are the deterministic backward's arithmetic,
    each chunk's FP32 weight-gradient partial is the deterministic backward's,
    and the partials are added exactly as 64-bit integer windows in the
    workspace; the CTA that arrives last rounds the exact sum once into ``dw``
    (adding ``grad_weight``'s value first when accumulating; FP32, then BF16
    for a BF16 ``dw``) and zeroes the workspace.  Every output repeats bit for
    bit.  Every call with rows uses a workspace, also with
    ``compute_grad_weight=False``, but one of up to 256 rows with a BF16
    weight gradient or with ``compute_grad_weight=False``: there one cluster
    of 16 CTAs runs (summing a weight gradient on chip in a fixed order) and
    uses none.  The arguments are those of
    :func:`flashinfer.cake_rmsnorm_silu_gate_nondet_train.cake_rmsnorm_silu_gate_nondet_train_backward`;
    ``workspace`` is this package's uint8 buffer
    (:func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`) or ``None`` (the
    buffer of the current device and stream; during CUDA Graph capture, a
    zero-filled buffer of the capture and capture stream).  No call raises
    PyTorch's alert for an operation without a deterministic implementation.
    """

    return _backward(
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


class CakeRMSNormSiLUGateNondetFunction(torch.autograd.Function):
    """Autograd of :func:`cake_rmsnorm_silu_gate_nondet` in this package: saves ``x``,
    ``z`` and ``weight`` (no FP32 intermediate) and recomputes the rest in the
    backward, so ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only
    the inputs as saved state.

    ``forward(x, z, weight, eps)``: both directions run this package's
    kernels, and every output repeats bit for bit; a checkpoint replay reruns
    the forward (the same ``y``).
    """

    @staticmethod
    def forward(ctx, x, z, weight, eps):
        y = _forward(x, z, weight, eps, out=None)
        ctx.save_for_backward(x, z, weight)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, z, weight = ctx.saved_tensors
        dx, dz, dw = _backward(
            grad_y,
            x,
            z,
            weight,
            ctx.eps,
            grad_x=None,
            grad_z=None,
            grad_weight=None,
            accumulate_grad_weight=False,
            compute_grad_weight=ctx.needs_input_grad[2],
            workspace=None,
        )
        return (
            dx if ctx.needs_input_grad[0] else None,
            dz if ctx.needs_input_grad[1] else None,
            dw,
            None,
        )


def cake_rmsnorm_silu_gate_nondet(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable ``RMSNorm(x) * weight * SiLU(z)`` over the last dimension
    (128) on this package's kernels: :class:`CakeRMSNormSiLUGateNondetFunction`
    applied to the arguments.
    """

    return CakeRMSNormSiLUGateNondetFunction.apply(x, z, weight, eps)


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
