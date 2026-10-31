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
its bits are the same); the backward is one kernel that writes ``dx`` and ``dz``
exactly as the deterministic backward does and adds each CTA's FP64
weight-gradient column sums into FP64 accumulators of the workspace with
atomics; the CTA that arrives last rounds the sum once into ``dw`` and zeroes
the accumulators.  The order of the FP64 additions depends on scheduling, so
``dw`` can move by one rounding of its dtype from call to call.

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
  the accumulators, then one element whose first four bytes are the 32-bit
  arrival counter (``4 * 128 + 1`` elements, 4104 bytes, for every shipped
  backward module).  The kernel leaves all of it zero, so a workspace zeroed
  once serves every later call on its stream, of any shape and either
  weight-gradient dtype, and every CUDA Graph replay, with no host write.  A
  caller workspace must come from
  :func:`cake_rmsnorm_silu_gate_train_backward_workspace` (zero-filled), or be
  zeroed with :func:`cake_rmsnorm_silu_gate_train_reset_workspace` after
  anything else wrote into it (a launch that did not complete, another use of
  the buffer); the kernel never needs a reset between its own calls.
- Retained state (FlashInfer issue #7039): the autograd function saves only
  ``x``, ``z`` and ``weight`` through ``save_for_backward``, so a non-reentrant
  checkpointed forward keeps nothing but its output.  A backward without a
  caller workspace uses the zero-filled workspace of its (device, stream,
  backward module), allocated on first use and kept in ``_STREAM_WORKSPACES``
  for later calls: 4104 bytes each, at most one per weight-gradient dtype (two
  backward modules) for every stream that ran such a backward.  A call during
  CUDA Graph capture allocates its own workspace instead, owned by the graph.
- No call synchronizes the host or the device; every kernel runs on the
  current torch stream with a plain (non-persistent, non-cooperative) grid.
"""

from __future__ import annotations

import functools
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
    ROLES,
    ROUTES,
    arch_for_capability,
    backward_chunks,
    route_module,
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


@functools.cache
def _sm_count_of_index(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


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
    return arch is not None and set(ROUTES.get(arch, {})) == set(ROLES)


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
    _loader.run_module(
        route_module(arch, "fwd"), values, rows=rows, device_index=_device_index(device)
    )
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
    reduction: the same bits).  The arguments are those of
    :func:`flashinfer.cake_rmsnorm_silu_gate_train.cake_rmsnorm_silu_gate_train_forward`;
    ``deterministic`` must be ``False``.
    """

    _require_nondeterministic(deterministic, "cake_rmsnorm_silu_gate_train_forward")
    return _forward(x, z, weight, eps, out=out)


def _workspace_elements(arch: str) -> int:
    """FP64 elements of a workspace that serves both weight-gradient dtypes."""

    return max(
        workspace_elements(route_module(arch, role)) for role in ("bwd_bf16", "bwd_f32")
    )


def cake_rmsnorm_silu_gate_train_backward_workspace_bytes(
    rows: int, device: torch.device, *, deterministic: bool = True
) -> int:
    """Bytes of the caller-owned backward workspace for ``rows`` rows
    (``T * H``) on ``device``: the FP64 weight-gradient accumulators and the
    32-bit arrival counter, a few KiB whatever ``rows``.  ``deterministic`` must
    be ``False``.
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
    FP64 accumulators and its 32-bit arrival counter.

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


# The workspaces of calls without a caller workspace: one per (device, stream,
# backward module), zero-filled once and kept for the process (4104 bytes each,
# at most two per stream).  Every backward leaves its workspace zero and the
# calls of one stream run in order, so they share it (no allocation and no fill
# per call).  A call during CUDA Graph capture allocates its own instead, so a
# graph replayed on another stream never shares a buffer with eager calls.
_STREAM_WORKSPACES: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _stream_workspace_views(
    name: str, arch: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(FP64 accumulators, uint32 arrival counter)`` of backward module
    ``name`` for a call without a caller workspace."""

    if torch.cuda.is_current_stream_capturing():
        return _workspace_views(
            torch.zeros(_workspace_elements(arch), dtype=torch.float64, device=device),
            name,
            device,
        )
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
    """``(FP64 accumulators, uint32 arrival counter)`` of backward module
    ``name`` on a caller's workspace."""

    elements = workspace_elements(name)
    flat = _check_workspace(workspace, device, elements)
    sums = flat[: elements - 1]
    done = flat[elements - 1 : elements].view(torch.uint32)[:1]
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
    name = route_module(
        arch, "bwd_f32" if dw is not None and dw.dtype == torch.float32 else "bwd_bf16"
    )
    chunks, blocks_per_cta = backward_chunks(
        name, rows, _sm_count_of_index(_device_index(device))
    )
    if workspace is not None:
        sums, done = _workspace_views(workspace, name, device)
    elif dw is not None:
        sums, done = _stream_workspace_views(name, arch, device)
    else:
        # weight_grad=0: the kernel reads and writes none of these.
        sums = torch.empty((0,), dtype=torch.float64, device=device)
        done = torch.empty((0,), dtype=torch.uint32, device=device)
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
        "blocks_per_cta": blocks_per_cta,
        "chunks": chunks,
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

    One kernel: ``dx`` and ``dz`` are the deterministic backward's bits, and each
    CTA adds its FP64 column sums into the FP64 accumulators of the workspace
    with atomics; the CTA that arrives last rounds the sum once into ``dw``
    (adding ``grad_weight``'s value first when accumulating) and zeroes the
    workspace.  The arguments are those of
    :func:`flashinfer.cake_rmsnorm_silu_gate_train.cake_rmsnorm_silu_gate_train_backward`;
    ``workspace`` is this package's FP64 buffer
    (:func:`cake_rmsnorm_silu_gate_train_backward_workspace`) or ``None`` (the
    buffer of the current device and stream) and ``deterministic`` must be
    ``False``.  Under ``torch.use_deterministic_algorithms(True)`` a call with
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
