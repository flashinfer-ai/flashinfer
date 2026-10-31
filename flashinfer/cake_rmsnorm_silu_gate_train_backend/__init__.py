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

Backend of :mod:`flashinfer.cake_rmsnorm_silu_gate_train`: the Cake SiLU-gated
RMSNorm training forward / backward (SM100 / SM103 / SM107) with a
deterministic default and an opt-in non-deterministic weight gradient, as one
self-contained package (host API bodies, the ``cake_jit`` registry, the pinned
``cake_build`` loader and the generated CUDA sources under ``csrc/``).  The
``@flashinfer_api``-decorated entry points and the user-facing documentation
live in :mod:`flashinfer.cake_rmsnorm_silu_gate_train`; import that module.

Every entry point takes the keyword ``deterministic`` (default ``True``):

- ``True`` calls the deterministic package
  :mod:`flashinfer.cake_rmsnorm_silu_gate_det_train_backend` with the same
  arguments, so the default call returns that package's bits and raises its
  errors.
- ``False`` runs this package's kernels: the forward is the deterministic
  package's forward kernel rendered here (it has no reduction, so its bits are
  the same); the backward is one kernel that writes ``dx`` and ``dz`` exactly as
  the deterministic backward does and adds each CTA's FP64 weight-gradient
  column sums into FP64 accumulators of the workspace with atomics; the CTA
  that arrives last rounds the sum once into ``dw`` and zeroes the
  accumulators.  The order of the FP64 additions depends on scheduling, so
  ``dw`` can move by one rounding of its dtype from call to call.
- ``False`` while ``torch.use_deterministic_algorithms(True)`` is in effect
  raises ``RuntimeError``; with ``warn_only=True`` it warns and runs the
  deterministic package.  A call that asks for determinism never runs the
  non-deterministic kernels.

Host design of the ``deterministic=False`` path (the input layout rules are the
deterministic package's):

- ``x``, ``z`` and the upstream gradient are BF16 tensors of one shape
  ``[..., 128]`` (``[T, H, 128]``, the flattened ``[T * H, 128]`` or any other
  leading shape), each with its own token and head strides; a layout the
  kernels cannot read is materialized once as a contiguous copy.
- The backward's workspace is ``cake_jit.workspace_elements`` FP64 elements:
  the accumulators, then one element whose first four bytes are the 32-bit
  arrival counter.  The kernel leaves all of it zero, so a workspace zeroed once
  serves every later call on its stream, of any shape and either weight-gradient
  dtype, and every CUDA Graph replay, with no host write.
- No call synchronizes the host or the device; every kernel runs on the
  current torch stream with a plain (non-persistent, non-cooperative) grid.
"""

from __future__ import annotations

import functools
import warnings
from typing import Optional

import torch

from .. import cake_rmsnorm_silu_gate_det_train_backend as _deterministic_backend
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

# The deterministic entry point each entry point of this package delegates to.
_DETERMINISTIC_NAMES = {
    "cake_rmsnorm_silu_gate_train_forward": "cake_rmsnorm_silu_gate_det_train_forward",
    "cake_rmsnorm_silu_gate_train_backward": "cake_rmsnorm_silu_gate_det_train_backward",
    "cake_rmsnorm_silu_gate_train_backward_workspace": "cake_rmsnorm_silu_gate_det_train_backward_workspace",
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes": "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate": "cake_rmsnorm_silu_gate_det",
}


def _resolve_deterministic(deterministic: bool, api: str) -> bool:
    """Whether a call of ``api`` runs the deterministic package.

    ``deterministic=True`` does.  ``deterministic=False`` runs this package's
    kernels unless ``torch.use_deterministic_algorithms(True)`` is in effect:
    then the call raises ``RuntimeError``, or, with ``warn_only=True``, warns
    and runs the deterministic package.
    """

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
    """Whether ``device`` is a CUDA device of an exported architecture.

    ``deterministic=True`` answers for the deterministic package
    (:func:`flashinfer.cake_rmsnorm_silu_gate_det_train.is_cake_rmsnorm_silu_gate_det_train_supported`),
    ``False`` for this package's routes.  The answer does not depend on
    ``torch.use_deterministic_algorithms``.
    """

    if deterministic:
        return _deterministic_backend.is_cake_rmsnorm_silu_gate_det_train_supported(
            device
        )
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
        with these arguments.  ``False`` runs this package's forward program, the
        same kernel rendered here (the forward has no reduction, so both return
        the same bits); it is refused while
        ``torch.use_deterministic_algorithms(True)`` is in effect (with
        ``warn_only=True``: a warning, then the deterministic forward).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape (``out`` when given).
    """

    if _resolve_deterministic(deterministic, "cake_rmsnorm_silu_gate_train_forward"):
        return _deterministic_backend.cake_rmsnorm_silu_gate_det_train_forward(
            x, z, weight, eps, out=out
        )
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
    (``T * H``) on ``device``.

    ``deterministic`` is resolved as the backward resolves it (see
    :func:`cake_rmsnorm_silu_gate_train_backward`), so the size fits the kernel
    the backward call with the same flag runs.  ``True``: the deterministic
    package's FP32 partial buffer, which grows with ``rows``
    (:func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes`).
    ``False``: the FP64 weight-gradient accumulators and the 32-bit arrival
    counter, a few KiB whatever ``rows``.
    """

    if _resolve_deterministic(
        deterministic, "cake_rmsnorm_silu_gate_train_backward_workspace_bytes"
    ):
        return _deterministic_backend.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
            rows, device
        )
    if int(rows) < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    return _workspace_elements(_device_arch(torch.device(device))) * 8


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
    """

    if _resolve_deterministic(
        deterministic, "cake_rmsnorm_silu_gate_train_backward_workspace"
    ):
        return (
            _deterministic_backend.cake_rmsnorm_silu_gate_det_train_backward_workspace(
                rows, device
            )
        )
    device = torch.device(device)
    if int(rows) < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    return torch.zeros(
        _workspace_elements(_device_arch(device)), dtype=torch.float64, device=device
    )


# The workspaces of calls without a caller workspace: one per (device, stream),
# zero-filled once.  Every non-deterministic backward leaves its workspace zero
# and the calls of one stream run in order, so they share it (no allocation and
# no fill per call).  A call during CUDA Graph capture allocates its own instead,
# so a graph replayed on another stream never shares a buffer with eager calls.
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


def _workspace_views(
    workspace: torch.Tensor, name: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(FP64 accumulators, uint32 arrival counter)`` of backward module
    ``name`` on a caller's workspace."""

    elements = workspace_elements(name)
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
            "(cake_rmsnorm_silu_gate_train_backward_workspace(..., deterministic=False))"
        )
    flat = workspace.view(-1)
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
        nor the workspace.
    workspace : torch.Tensor, optional
        Caller-owned buffer from :func:`cake_rmsnorm_silu_gate_train_backward_workspace`
        called with the same ``deterministic`` flag: the deterministic
        package's FP32 partial buffer (sized for at least this many rows), or the
        zero-filled FP64 buffer of the non-deterministic backward, which the
        kernel leaves zero.  When omitted: the deterministic package allocates
        its buffer per call; the non-deterministic backward uses one zero-filled
        buffer per device and stream, kept for later calls (a call during CUDA
        Graph capture allocates its own).  One buffer per backward that may run
        at the same time on another stream.
    deterministic : bool
        ``True`` (default) runs the deterministic package's backward
        (:func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward`)
        with these arguments: ``dw`` accumulated in FP32 per CTA chunk and in
        FP64 over the chunks in an order fixed by the row count, rounded once,
        so repeated calls return the same bits.  ``False`` runs one kernel: the
        same ``dx`` and ``dz`` bits, and each CTA adds its FP64 column sums of
        the same FP32 runs into FP64 accumulators with atomics; the CTA that
        arrives last rounds the sum once into ``dw`` (adding ``grad_weight``'s
        value first when accumulating).  The order of those FP64 additions
        depends on scheduling, so ``dw`` can differ by one rounding of its dtype
        between calls; its error bound is the deterministic backward's.
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
        return _deterministic_backend.cake_rmsnorm_silu_gate_det_train_backward(
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
    """Autograd of :func:`cake_rmsnorm_silu_gate`: saves ``x``, ``z`` and
    ``weight`` (no FP32 intermediate) and recomputes the rest in the backward,
    so ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only the inputs
    as saved state.

    ``forward(x, z, weight, eps, deterministic=True)``: ``deterministic``
    selects the package of both directions (see
    :func:`cake_rmsnorm_silu_gate_train_backward`).  The forward resolves it
    (refusing ``False`` under ``torch.use_deterministic_algorithms(True)``
    before any launch) and the backward resolves the same flag again when it
    runs; a checkpoint replay reruns the forward with the same flag.
    """

    @staticmethod
    def forward(ctx, x, z, weight, eps, deterministic=True):
        if _resolve_deterministic(deterministic, "cake_rmsnorm_silu_gate"):
            y = _deterministic_backend.cake_rmsnorm_silu_gate_det_train_forward(
                x, z, weight, eps
            )
        else:
            y = _forward(x, z, weight, eps, out=None)
        ctx.save_for_backward(x, z, weight)
        ctx.eps = float(eps)
        ctx.deterministic = bool(deterministic)
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
            deterministic=ctx.deterministic,
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
    (128): :class:`CakeRMSNormSiLUGateFunction` applied to the arguments.

    ``deterministic=True`` (default) runs the deterministic package in both
    directions, the bits of
    :func:`flashinfer.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det`.
    ``False`` runs this package's kernels, whose weight gradient is summed over
    the CTAs in a scheduling-dependent order; it is refused while
    ``torch.use_deterministic_algorithms(True)`` is in effect (with
    ``warn_only=True``: a warning, then the deterministic kernels).
    """

    return CakeRMSNormSiLUGateFunction.apply(x, z, weight, eps, deterministic)


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
