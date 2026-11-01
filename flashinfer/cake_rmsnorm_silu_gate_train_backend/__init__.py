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

Non-deterministic backend of :mod:`flashinfer.cake_rmsnorm_silu_gate_train`: the
Cake SiLU-gated RMSNorm training forward / backward (SM100 / SM103 / SM107) with
the opt-in non-deterministic weight gradient, as one self-contained package (host
API bodies, the ``cake_jit`` registry, the pinned ``cake_build`` loader and the
generated CUDA sources under ``csrc/``).  It imports nothing from other
flashinfer packages and ships its own copy of every program it launches.  The
dispatching entry points (``deterministic=True`` -> the deterministic package
``flashinfer.cake_rmsnorm_silu_gate_det_train``), the ``@flashinfer_api``
decorations and the user-facing documentation live in
:mod:`flashinfer.cake_rmsnorm_silu_gate_train`; import that module.

Numerical contract against the deterministic package: the forward is the
deterministic package's forward kernel rendered here (it has no reduction, so
its bits are the same), and on sm_100a / sm_103a from 976 rows per SM a
forward whose CTAs each walk several row blocks with the same arithmetic (the
same bits); the backward is one kernel that writes ``dx`` and ``dz``
exactly as the deterministic backward does, sums the weight gradient of the
row blocks each CTA takes (its own block when the grid has a CTA for every
block, else the chunks it claims) in FP32 runs of at most 16 rows folded into
FP64 (a CTA of a few blocks adds its runs across its warps in FP32 and
converts once: at most 16 FP32 roundings a term either way), and adds each
CTA's FP64 column sums into
FP64 accumulators of the workspace with atomics; the CTA that arrives last
rounds the sum once into ``dw`` and zeroes the workspace.  The claimed blocks,
and so the FP32 runs, and the order of the FP64 additions depend on
scheduling: ``dw`` can move from call to call, always within the deterministic
backward's error bound ``|dw64| * 2^-7 + sum|terms| * 2^-20`` per column.

Numerical-mode keyword: every compute and workspace entry point (and the
forward of :class:`CakeRMSNormSiLUGateFunction`) takes
``deterministic: bool = True`` and raises ``ValueError`` unless it is
``False``: this package has no deterministic backward, so a call must opt in.
The support query takes the keyword too but answers for any value (it launches
nothing).
With ``torch.use_deterministic_algorithms(True)`` the paths that launch the
unordered weight-gradient sum -- :func:`cake_rmsnorm_silu_gate_train_backward`
with ``compute_grad_weight=True`` and the backward of the autograd function when
``weight`` needs a gradient -- raise PyTorch's alert for an operation without a
deterministic implementation (a ``RuntimeError``, or a ``UserWarning`` under
``warn_only=True``, after which the non-deterministic backward runs) before any
validation or launch, and never fall back.  The forward, the workspace entry
points and the support query do not alert.

Host design (the input layout rules are the deterministic package's):

- ``x``, ``z`` and the upstream gradient are BF16 tensors of one shape
  ``[..., 128]`` (``[T, H, 128]``, the flattened ``[T * H, 128]`` or any other
  leading shape), each with its own token and head strides; a layout the
  kernels cannot read is materialized once as a contiguous copy.
- The backward's workspace is ``cake_jit.workspace_elements`` FP64 elements:
  the accumulators, then one 8-byte element holding the two 32-bit words of
  ``dw_done``, its first four bytes the arrival ticket and its last four the
  claim counter: ``4 * 128 + 1`` elements, 4104 bytes, for every shipped
  backward module.  Every backward launch takes a ticket (and past one wave
  of row blocks claims its blocks from the counter), also one with
  ``compute_grad_weight=False``, so every backward launch uses a workspace.
  The kernel leaves all of it zero, so a workspace zeroed once serves every
  later call on its stream, of any shape and either weight-gradient dtype, and
  every CUDA Graph replay, with no host write.  A caller workspace must come
  from :func:`cake_rmsnorm_silu_gate_train_backward_workspace` (zero-filled),
  or be zeroed with :func:`cake_rmsnorm_silu_gate_train_reset_workspace` after
  anything else wrote into it (a launch that did not complete, another use of
  the buffer); the kernel never needs a reset between its own calls.
- Retained state (FlashInfer issue #7039): the autograd function saves only
  ``x``, ``z`` and ``weight`` through ``save_for_backward``, so a non-reentrant
  checkpointed forward keeps nothing but its output.  A backward without a
  caller workspace (``compute_grad_weight=False`` included: it runs the BF16
  weight-gradient module) uses the zero-filled workspace of its (device,
  stream, backward module), allocated on first use and kept in
  ``_STREAM_WORKSPACES`` for later calls: 4104 bytes each, at most one per
  backward module (two per weight-gradient dtype on sm_107a, where the long
  backward is a module of its own, else one) for every stream that ran such a
  backward.  Such eager calls also keep a reserve of 64 zero-filled
  workspaces per device (one allocation of about 272 KiB, filled outside any
  capture), from which a call during CUDA Graph capture takes the workspace of
  its capture and capture stream for good, so a replay launches no fill kernel
  (see ``_RESERVES``; with none ready, the graph zero-fills one of its own).
- Grids (``cake_jit.module_for_rows`` picks the module from the rows and
  the device's SM count): the forward launches one CTA per block of 16 rows;
  on sm_100a and sm_103a it launches a forward whose CTA ``b`` walks the
  blocks ``b``, ``b + grid``, ..., loading each next block while it computes
  the current one, in the fewest whole waves that give no CTA more than a
  few blocks: from 976 rows per SM the mid forward (6 CTAs per SM, at most 2
  blocks of 16 rows each), from 2400 the long forward (3 CTAs per SM, at most
  6 blocks of 32 rows each).  The backward
  launches ``cake_jit.backward_ctas`` CTAs, ``G = min(row blocks,
  ctas_per_sm * SMs)``, so at most one wave: 256-thread CTAs of 16-row blocks
  with ``ctas_per_sm = 4``, and on sm_107a from 1024 rows per SM 128-thread
  CTAs of 8-row blocks with ``ctas_per_sm = 8``.  When
  ``G`` is the number of row blocks, CTA ``c`` takes block ``c`` and nothing
  is claimed.  Past one wave, thread 0 of every CTA claims every block in
  chunks of consecutive blocks with a relaxed atomic increment on the
  workspace's claim counter, from a table of halving rounds that ends in
  one-block chunks: its first chunk as the CTA starts and each next one as the
  last block of its current chunk starts, so a CTA holds one chunk at a time,
  a CTA that starts late takes fewer blocks, and a CTA that becomes resident
  after the table ran out leaves without a row.  Each thread folds its FP32
  sums into FP64 after at most 16 of its rows (a CTA of a few blocks adds them
  across its warps in FP32 first and converts once).  No kernel is
  cooperative and no CTA waits for another; no call synchronizes the host or
  the device, and every kernel runs on the current torch stream.
- Beside kernels resident on other streams (FlashInfer issue #7548; observed
  on B200, sm_100a): each 256-thread backward CTA needs 20736 bytes of dynamic
  shared memory (each 128-thread CTA of the long backward on sm_107a 10496), so
  a kernel holding every SM with a small shared-memory carveout delays the
  backward's start until it exits;
  :mod:`flashinfer.cake_rmsnorm_silu_gate_train` documents the remedy.
"""

from __future__ import annotations

import functools
import threading
import warnings
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
    backward_ctas,
    device_sm_count,
    module_for_rows,
    workspace_elements,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
_ALIGNMENT_BYTES = ALIGNMENT_ELEMENTS * 2


def _require_nondeterministic(deterministic: Any, entry: str) -> None:
    """The mode contract of every entry point: this package sums the weight
    gradient non-deterministically, so an entry point takes
    ``deterministic=False`` and raises ``ValueError`` for any other value (the
    deterministic package is ``flashinfer.cake_rmsnorm_silu_gate_det_train``)."""

    if deterministic is not False:
        raise ValueError(
            f"{entry} computes non-deterministically and takes deterministic=False "
            f"(got {deterministic!r}); flashinfer.cake_rmsnorm_silu_gate_det_train "
            "is the deterministic package"
        )


def _alert_nondeterministic(entry: str) -> None:
    """PyTorch's alert for an operation without a deterministic implementation,
    raised by ``entry`` before it launches the unordered weight-gradient sum
    while PyTorch's deterministic algorithms are enabled: a ``RuntimeError``, or
    a ``UserWarning`` under ``torch.use_deterministic_algorithms(True,
    warn_only=True)`` (after which the call runs).  Launches without that sum do
    not alert."""

    if not torch.are_deterministic_algorithms_enabled():
        return
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{entry} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'; "
            "flashinfer.cake_rmsnorm_silu_gate_det_train is the deterministic package",
            UserWarning,
            stacklevel=3,
        )
        return
    raise RuntimeError(
        f"{entry} does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True)'; "
        "flashinfer.cake_rmsnorm_silu_gate_det_train is the deterministic package, "
        "or use the 'warn_only=True' option if that is acceptable for your "
        "application"
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


def is_cake_rmsnorm_silu_gate_train_supported(
    device: torch.device, *, deterministic: bool = True
) -> bool:
    """Whether ``device`` is a CUDA device with every route of this package.

    A capability probe that launches nothing: it never raises for
    ``deterministic`` (the answer is this package's for any value), and does
    not depend on ``torch.use_deterministic_algorithms``.
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
    A stride of a size-1 dimension is never dereferenced and is passed as 0.
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
        head = int(view.stride(1)) if heads > 1 else 0
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
        HEAD_DIM if heads > 1 else 0,
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
    """The forward of this package's program (``deterministic=False``)."""

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


def cake_rmsnorm_silu_gate_train_forward(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    r"""SiLU-gated RMSNorm forward of this package:
    ``y = BF16(((x * rstd) * weight) * silu(z))``.

    The deterministic package's forward kernel rendered here (it has no
    reduction: the same bits), or on sm_100a / sm_103a from 976 rows per SM
    the mid forward and from 2400 the long forward (the same bits).  The
    arguments are those of
    :func:`flashinfer.cake_rmsnorm_silu_gate_train.cake_rmsnorm_silu_gate_train_forward`;
    ``deterministic`` must be ``False``.
    """

    _require_nondeterministic(deterministic, "cake_rmsnorm_silu_gate_train_forward")
    return _forward(x, z, weight, eps, out=out)


def _workspace_elements(arch: str) -> int:
    """FP64 elements of a workspace that serves every backward module of
    ``arch`` (both weight-gradient dtypes, any row count)."""

    routes = ROUTES[arch]
    return max(
        workspace_elements(routes[role])
        for role in ROLES + LONG_ROLES
        if role.startswith("bwd") and role in routes
    )


def cake_rmsnorm_silu_gate_train_backward_workspace_bytes(
    rows: int, device: torch.device, *, deterministic: bool = True
) -> int:
    """Bytes of the caller-owned backward workspace for ``rows`` rows
    (``T * H``) on ``device``: the FP64 weight-gradient accumulators and the
    8-byte word of the arrival ticket and the claim counter, a few KiB whatever
    ``rows``.  ``deterministic`` must be ``False``.
    """

    _require_nondeterministic(
        deterministic, "cake_rmsnorm_silu_gate_train_backward_workspace_bytes"
    )
    if int(rows) < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    return _workspace_elements(_device_arch(torch.device(device))) * 8


def cake_rmsnorm_silu_gate_train_backward_workspace(
    rows: int, device: torch.device, *, deterministic: bool = True
) -> torch.Tensor:
    """A caller-owned backward workspace for ``rows`` rows (``T * H``): a
    zero-filled FP64 tensor of
    :func:`cake_rmsnorm_silu_gate_train_backward_workspace_bytes` bytes.

    The backward leaves it zero after every call, so it serves every later call
    on its stream (any ``T`` and ``H``, BF16 or FP32 weight gradient) and every
    CUDA Graph replay without a reset.  A backward that may run at the same time
    on another stream needs its own workspace.  ``deterministic`` must be
    ``False``.
    """

    _require_nondeterministic(
        deterministic, "cake_rmsnorm_silu_gate_train_backward_workspace"
    )
    device = torch.device(device)
    if int(rows) < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    return torch.zeros(
        _workspace_elements(_device_arch(device)), dtype=torch.float64, device=device
    )


def cake_rmsnorm_silu_gate_train_reset_workspace(
    workspace: torch.Tensor, *, deterministic: bool = True
) -> None:
    """Return a caller-owned backward workspace to its first-use state: zero its
    FP64 accumulators and the two 32-bit words of its last element (the arrival
    ticket and the claim counter).

    Every backward leaves its workspace zero, so reuse across calls, shapes,
    weight-gradient dtypes and CUDA Graph replays needs no reset.  A workspace
    that did not come from :func:`cake_rmsnorm_silu_gate_train_backward_workspace`,
    or that something else wrote into (a launch that did not complete, another
    use of the buffer), must be reset before its next backward.  The fill runs
    on the current stream of the workspace's device, without a host
    synchronization (it can be captured in a CUDA Graph).  The workspace must be
    a contiguous FP64 CUDA tensor of at least
    :func:`cake_rmsnorm_silu_gate_train_backward_workspace_bytes` bytes; all of
    it is zeroed.  ``deterministic`` must be ``False``.
    """

    _require_nondeterministic(
        deterministic, "cake_rmsnorm_silu_gate_train_reset_workspace"
    )
    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace must be a torch.Tensor")
    device = workspace.device
    _check_workspace(workspace, device, _workspace_elements(_device_arch(device)))
    workspace.zero_()


# The workspaces of calls without a caller workspace (every backward launch
# needs one: it takes the arrival ticket, and claims row blocks past one
# wave): one
# per (device, stream, backward module), zero-filled once and kept for the
# process (4104 bytes each, at most two per stream, four on sm_107a).  Every
# backward leaves its workspace zero and the calls of one stream run in order,
# so they share it (no allocation and no fill per call).  A call during CUDA Graph capture uses a
# workspace of its capture instead (``_capture_workspace``), so a graph replayed
# on another stream never shares a buffer with eager calls or other graphs.
_STREAM_WORKSPACES: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

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
        elements = _workspace_elements(arch)
        stride = -(-elements // 32) * 32  # 256-byte aligned workspaces
        block = torch.zeros(count * stride, dtype=torch.float64, device=device)
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
    """The FP64 workspace of a call without a caller workspace during CUDA
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
            _workspace_elements(arch), dtype=torch.float64, device=device
        )
    if capture is not None:
        _CAPTURE_WORKSPACES[key] = (capture, workspace)
    return workspace


def _stream_workspace_views(
    name: str, arch: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(FP64 accumulators, uint32 dw_done words)`` of backward module
    ``name`` for a call without a caller workspace."""

    if torch.cuda.is_current_stream_capturing():
        return _workspace_views(_capture_workspace(arch, device), name, device)
    _refill_reserve(arch, device)
    key = (_device_index(device), torch.cuda.current_stream(device).cuda_stream, name)
    views = _STREAM_WORKSPACES.get(key)
    if views is None:
        views = _workspace_views(
            torch.zeros(_workspace_elements(arch), dtype=torch.float64, device=device),
            name,
            device,
        )
        _STREAM_WORKSPACES[key] = views
    return views


def _check_workspace(
    workspace: torch.Tensor, device: torch.device, elements: int
) -> torch.Tensor:
    """The flat view of a caller's workspace of at least ``elements`` FP64
    elements on ``device``."""

    if (
        not isinstance(workspace, torch.Tensor)
        or workspace.device != device
        or workspace.dtype != torch.float64
        or not workspace.is_contiguous()
        or workspace.numel() < elements
    ):
        raise ValueError(
            f"workspace must be a contiguous FP64 tensor on {device} with at least "
            f"{elements} elements, zero before its first use "
            "(cake_rmsnorm_silu_gate_train_backward_workspace(..., deterministic=False), "
            "or cake_rmsnorm_silu_gate_train_reset_workspace(..., deterministic=False))"
        )
    return workspace.view(-1)


def _workspace_views(
    workspace: torch.Tensor, name: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(FP64 accumulators, uint32 dw_done words)`` of backward module
    ``name`` on a caller's workspace: ``dw_done`` views both 32-bit words of
    the last element, ``[0]`` the arrival ticket and ``[1]`` the claim
    counter."""

    elements = workspace_elements(name)
    flat = _check_workspace(workspace, device, elements)
    sums = flat[: elements - 1]
    done = flat[elements - 1 : elements].view(torch.uint32)
    return sums, done


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
    """The backward of this package's program (``deterministic=False``)."""

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
    # Every launch, weight_grad=0 included, takes the arrival ticket (and past
    # one wave of row blocks claims its blocks from the claim counter), and
    # leaves both words zero.
    if workspace is not None:
        sums, done = _workspace_views(workspace, name, device)
    else:
        sums, done = _stream_workspace_views(name, arch, device)
    values = {
        "dy": gv,
        "x": xv,
        "z": zv,
        "w": w,
        "dx": dx,
        "dz": dz,
        "dw_sums": sums,
        "dw_done": done,
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
    r"""SiLU-gated RMSNorm backward of this package: ``dx``, ``dz`` and the
    weight gradient.

    One kernel of at most one wave of CTAs (one row block each when the grid
    has a CTA for every block; past one wave they claim the blocks in chunks
    from the workspace's claim counter): ``dx`` and ``dz`` are
    the deterministic backward's bits, and each CTA adds its FP64 column sums
    into the FP64 accumulators of the workspace with atomics; the CTA that
    arrives last rounds the sum once into ``dw`` (adding ``grad_weight``'s value
    first when accumulating) and zeroes the workspace.  Every call with rows
    uses a workspace, also with ``compute_grad_weight=False``.  The arguments
    are those of
    :func:`flashinfer.cake_rmsnorm_silu_gate_train.cake_rmsnorm_silu_gate_train_backward`;
    ``workspace`` is this package's FP64 buffer
    (:func:`cake_rmsnorm_silu_gate_train_backward_workspace`) or ``None`` (the
    buffer of the current device and stream; during CUDA Graph capture, a
    zero-filled buffer of the capture and capture stream) and
    ``deterministic`` must be ``False``.  Under
    ``torch.use_deterministic_algorithms(True)`` a call with
    ``compute_grad_weight=True`` raises PyTorch's alert (``RuntimeError``; a
    ``UserWarning`` under ``warn_only=True``, then it runs) before any
    validation or launch.
    """

    _require_nondeterministic(deterministic, "cake_rmsnorm_silu_gate_train_backward")
    if compute_grad_weight:
        _alert_nondeterministic("cake_rmsnorm_silu_gate_train_backward")
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


class CakeRMSNormSiLUGateFunction(torch.autograd.Function):
    """Autograd of :func:`cake_rmsnorm_silu_gate` in this package: saves ``x``,
    ``z`` and ``weight`` (no FP32 intermediate) and recomputes the rest in the
    backward, so ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only
    the inputs as saved state.

    ``forward(x, z, weight, eps, deterministic=True)``: ``deterministic`` must be
    ``False`` (any other value raises ``ValueError`` before any launch); both
    directions run this package's kernels.  The backward, when ``weight`` needs
    a gradient, raises PyTorch's alert if
    ``torch.use_deterministic_algorithms(True)`` is in effect when it runs (see
    :func:`cake_rmsnorm_silu_gate_train_backward`); a checkpoint replay reruns
    the forward.
    """

    @staticmethod
    def forward(ctx, x, z, weight, eps, deterministic=True):
        _require_nondeterministic(deterministic, "CakeRMSNormSiLUGateFunction")
        y = _forward(x, z, weight, eps, out=None)
        ctx.save_for_backward(x, z, weight)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, z, weight = ctx.saved_tensors
        dx, dz, dw = cake_rmsnorm_silu_gate_train_backward(
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


def cake_rmsnorm_silu_gate(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """Differentiable ``RMSNorm(x) * weight * SiLU(z)`` over the last dimension
    (128) on this package's kernels: :class:`CakeRMSNormSiLUGateFunction`
    applied to the arguments.  ``deterministic`` must be ``False``.
    """

    _require_nondeterministic(deterministic, "cake_rmsnorm_silu_gate")
    return CakeRMSNormSiLUGateFunction.apply(x, z, weight, eps, False)


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
