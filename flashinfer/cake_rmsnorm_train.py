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

Cake fused BF16 RMSNorm training forward / backward (SM100 / SM103 / SM107).

Forward: ``r_t = (mean_j x_tj^2 + eps)^(-1/2)`` in FP32, ``y = BF16((x * r) * w)``.
The only tensor saved for the backward besides the inputs is the FP32 per-row
reciprocal RMS ``rstd`` (``[T]``).  Backward: one fused pass produces
``dx = BF16(r * g * w - x * r^3 * mean(g * w * x))`` and the FP32 weight
gradient ``dw = sum_t g * x * r`` through a fixed row-chunk partition and a
fixed-order reduction, so ``dw`` is bitwise reproducible for a given token
count on a given architecture.  The optional residual-add fusion computes
``h_new = BF16(h + u)`` first, normalizes ``h_new``, and in the backward adds
the gradient flowing directly through ``h_new`` so that one ``dx`` tensor is
the gradient of both ``h`` and ``u``.

All kernels use FP32 statistics and accumulation with IEEE division and
square root (no fast-math).  Inputs and outputs are BF16; ``rstd`` and ``dw``
are FP32.  Row-strided inputs (``x``, ``residual``, ``g``, ``g_residual``)
are accepted when ``stride(1) == 1``, ``stride(0)`` is a multiple of eight
elements and the storage is aligned (32 bytes for the forward inputs, 16 for
the backward); outputs are contiguous.  The
token count ``T`` is a runtime scalar: it may change between calls, ``T == 1``
is supported and ``T == 0`` returns empty outputs without launching.

Usage::

    y, rstd = cake_rmsnorm_train_forward(x, w, eps)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd)          # dw is FP32 [H]

    # autograd (dw is returned in w.dtype; the kernel accumulates in FP32)
    y = cake_rmsnorm(x, w, eps)
    y, h_new = cake_rmsnorm(h, w, eps, residual=u)                 # fused add

    # caller-owned backward workspace (partials + self-resetting counters)
    ws = cake_rmsnorm_train_backward_workspace(T, H, x.device)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd, workspace=ws)
"""

from __future__ import annotations

from typing import Optional

import torch

from .api_logging import flashinfer_api
from .jit import cake_rmsnorm_train as _loader
from .jit.cake_rmsnorm_train import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    BACKWARD_ALIGNMENT_BYTES,
    FORWARD_ALIGNMENT_BYTES,
    arch_for_capability,
    route_applies,
    route_key,
    route_modules,
    selected_modules,
    supported_hidden_sizes,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"Cake RMSNorm training requires CUDA tensors, got {device}")
    capability = tuple(torch.cuda.get_device_capability(device))
    arch = arch_for_capability(capability)
    if arch is None:
        raise ValueError(
            "Cake RMSNorm training supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {capability}"
        )
    return arch


def is_cake_rmsnorm_train_supported(
    device: torch.device, hidden: Optional[int] = None
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has an exported route."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    arch = arch_for_capability(tuple(torch.cuda.get_device_capability(device)))
    if arch is None:
        return False
    if hidden is None:
        return bool(supported_hidden_sizes(arch))
    return int(hidden) in supported_hidden_sizes(arch)


def _validate_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _materialize(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    """A contiguous copy whose storage satisfies ``alignment`` (fresh allocations are 256-byte aligned)."""

    out = tensor.contiguous()
    if out.data_ptr() % alignment:
        out = out.clone(memory_format=torch.contiguous_format)
    return out


def _kernel_input(
    tensor: torch.Tensor,
    name: str,
    *,
    rows: int,
    hidden: int,
    device: torch.device,
    dtype: torch.dtype,
    alignment: int,
) -> torch.Tensor:
    """Validate a ``[rows, hidden]`` input of any layout and return what the kernel can read.

    The kernels read a row-strided tensor in place when the feature dimension is
    contiguous, the storage base is ``alignment``-byte aligned and the row stride
    in bytes is a multiple of ``alignment`` (``stride(0) == 0`` of an expanded
    gradient included).  Any other layout -- odd column offset, odd row stride,
    transposed storage -- is materialized once as a contiguous copy.  A single row
    never dereferences its row stride and is re-viewed with the canonical stride.
    """

    _validate_tensor(tensor, name, shape=(rows, hidden), device=device, dtype=dtype)
    if rows == 0 or hidden == 0:
        return tensor
    feature_contiguous = tensor.stride(1) == 1 or hidden == 1
    if feature_contiguous and tensor.data_ptr() % alignment == 0:
        if rows == 1:
            return (
                tensor
                if tensor.stride(0) == hidden
                else tensor.as_strided((1, hidden), (hidden, 1))
            )
        row_stride = tensor.stride(0)
        if (row_stride * tensor.element_size()) % alignment == 0 and (
            row_stride >= hidden or row_stride == 0
        ):
            return tensor
    return _materialize(tensor, alignment)


def _contiguous_input(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    alignment: int,
) -> torch.Tensor:
    """Validate a small input (``w``, ``rstd``); contiguous and aligned in place, else a copy."""

    _validate_tensor(tensor, name, shape=shape, device=device, dtype=dtype)
    if tensor.numel() == 0 or (
        tensor.is_contiguous() and tensor.data_ptr() % alignment == 0
    ):
        return tensor
    return _materialize(tensor, alignment)


def _output_buffer(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    alignment: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(kernel buffer, user buffer to copy back into or None)``.

    A caller-provided output of any layout is honoured: the kernel writes a
    contiguous aligned buffer directly when the caller's tensor is one, otherwise
    into a temporary that is copied back into the caller's view after the launch.
    """

    if tensor is None:
        return torch.empty(shape, dtype=dtype, device=device), None
    _validate_tensor(tensor, name, shape=shape, device=device, dtype=dtype)
    if tensor.numel() == 0 or (
        tensor.is_contiguous() and tensor.data_ptr() % alignment == 0
    ):
        return tensor, None
    return torch.empty(shape, dtype=dtype, device=device), tensor


def _copy_back(pairs: list[tuple[torch.Tensor, Optional[torch.Tensor]]]) -> None:
    for kernel_buffer, user_buffer in pairs:
        if user_buffer is not None:
            user_buffer.copy_(kernel_buffer)


def _backward_module_records(arch: str, hidden: int, residual: bool) -> list[dict]:
    return [
        _loader.MODULES[name] for name in route_modules(arch, hidden, "bwd", residual)
    ]


def _workspace_record(arch: str, hidden: int, residual: bool) -> dict:
    """The backward module that owns the partial / counter workspace."""

    owners = [
        record
        for record in _backward_module_records(arch, hidden, residual)
        if record.get("workspace_rule") is not None
    ]
    if len(owners) != 1:
        raise RuntimeError(
            f"the exported backward route for hidden={hidden} declares "
            f"{len(owners)} workspace owners; expected exactly one"
        )
    return owners[0]


def cake_rmsnorm_train_backward_workspace_bytes(
    rows: int, hidden: int, device: torch.device, *, residual: bool = False
) -> int:
    """Bytes of the caller-owned backward workspace of width ``hidden`` on ``device``.

    The workspace holds the FP32 per-chunk ``dw`` partials (one row per CTA of
    the persistent backward grid, sized for the device's largest chunk count
    so it is independent of ``rows``) followed by the ``uint32`` completion
    counters.  Counters must be zero on first use (allocate with
    ``torch.zeros`` or use :func:`cake_rmsnorm_train_backward_workspace`); the
    kernels leave them zero, so the same buffer is reusable across calls, any
    token count and CUDA Graph replays without host writes.
    """

    device = torch.device(device)
    rows = int(rows)
    hidden = int(hidden)
    if rows < 0:
        raise ValueError("rows must be non-negative")
    arch = _device_arch(device)
    record = _workspace_record(arch, hidden, residual)
    layout = _loader.workspace_layout(
        record, max(rows, 1), device_index=_device_index(device)
    )
    return int(layout["total_bytes"])


def cake_rmsnorm_train_backward_workspace(
    rows: int, hidden: int, device: torch.device, *, residual: bool = False
) -> torch.Tensor:
    """Allocate a zero-initialized backward workspace (see ``..._workspace_bytes``)."""

    nbytes = cake_rmsnorm_train_backward_workspace_bytes(
        rows, hidden, device, residual=residual
    )
    return torch.zeros(nbytes, dtype=torch.uint8, device=torch.device(device))


def _workspace_views(
    workspace: torch.Tensor, record: dict, *, rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    layout = _loader.workspace_layout(record, rows, device_index=_device_index(device))
    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace must be a torch.Tensor")
    if (
        workspace.device != device
        or workspace.dtype != torch.uint8
        or workspace.ndim != 1
    ):
        raise ValueError(
            f"workspace must be a 1-D uint8 CUDA tensor on {device}, got "
            f"{workspace.dtype} {tuple(workspace.shape)} on {workspace.device}"
        )
    if not workspace.is_contiguous() or workspace.data_ptr() % 256 != 0:
        raise ValueError("workspace must be contiguous and 256-byte aligned")
    if workspace.numel() < layout["total_bytes"]:
        raise ValueError(
            f"workspace holds {workspace.numel()} bytes; this call needs "
            f"{layout['total_bytes']} (cake_rmsnorm_train_backward_workspace_bytes)"
        )
    hidden = int(record["hidden"])
    partial = workspace[: 4 * layout["n_chunks"] * hidden].view(torch.float32)
    partial = partial.view(layout["n_chunks"], hidden)
    begin = layout["counters_offset"]
    counters = workspace[begin : begin + 4 * max(layout["counters"], 1)].view(
        torch.uint32
    )
    return partial, counters, layout


@flashinfer_api
def cake_rmsnorm_train_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    residual: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    r"""Fused BF16 RMSNorm forward with FP32 statistics.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, H]`` input of any layout.  Row-strided views with a contiguous
        feature dimension, 32-byte aligned storage and a row stride that is a
        multiple of 32 bytes are read in place; other layouts (odd column offset,
        odd row stride, transposed storage) are materialized once.  With
        ``residual`` this is the residual stream ``h`` and the normalized value is
        ``h_new = BF16(h + residual)``.
    w : torch.Tensor
        BF16 ``[H]`` weight (copied when not contiguous).
    eps : float
        Variance epsilon (FP32 inside the kernel).
    residual : torch.Tensor, optional
        BF16 ``[T, H]`` update added to ``x`` before normalization (same layout
        rules as ``x``).  Enables the fused residual-add route.
    out, rstd, residual_out : torch.Tensor, optional
        Preallocated outputs: BF16 ``[T, H]`` ``y``, FP32 ``[T]`` reciprocal RMS
        and (residual only) BF16 ``[T, H]`` ``h_new``.  Contiguous aligned buffers
        are written directly; any other layout receives a copy after the launch.

    Returns
    -------
    ``(y, rstd)`` or ``(y, rstd, h_new)`` when ``residual`` is given.
    ``rstd`` is the FP32 per-row ``1 / sqrt(mean(h_new^2) + eps)`` the backward
    needs; no other ``[T, H]`` temporary is produced.
    """

    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("x must be a [T, H] tensor")
    device = x.device
    rows, hidden = int(x.shape[0]), int(x.shape[1])
    arch = _device_arch(device)
    has_residual = residual is not None
    align = FORWARD_ALIGNMENT_BYTES
    x = _kernel_input(
        x,
        "x",
        rows=rows,
        hidden=hidden,
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
    )
    w = _contiguous_input(
        w,
        "w",
        shape=(hidden,),
        device=device,
        dtype=torch.bfloat16,
        alignment=BACKWARD_ALIGNMENT_BYTES,
    )
    if has_residual:
        residual = _kernel_input(
            residual,
            "residual",
            rows=rows,
            hidden=hidden,
            device=device,
            dtype=torch.bfloat16,
            alignment=align,
        )
    if not route_applies(
        device_capability=tuple(torch.cuda.get_device_capability(device)),
        hidden=hidden,
        kind="fwd",
        residual=has_residual,
    ):
        raise ValueError(
            f"no exported Cake RMSNorm training forward for hidden={hidden} "
            f"(residual={has_residual}) on {arch}; supported hidden sizes: "
            f"{supported_hidden_sizes(arch)}"
        )
    out, out_user = _output_buffer(
        out,
        "out",
        shape=(rows, hidden),
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
    )
    rstd, rstd_user = _output_buffer(
        rstd,
        "rstd",
        shape=(rows,),
        device=device,
        dtype=torch.float32,
        alignment=BACKWARD_ALIGNMENT_BYTES,
    )
    residual_out_user = None
    if has_residual:
        residual_out, residual_out_user = _output_buffer(
            residual_out,
            "residual_out",
            shape=(rows, hidden),
            device=device,
            dtype=torch.bfloat16,
            alignment=align,
        )
    copy_backs = [(out, out_user), (rstd, rstd_user), (residual_out, residual_out_user)]
    results = tuple(
        user if user is not None else kernel
        for kernel, user in copy_backs[: 3 if has_residual else 2]
    )
    if rows == 0:
        return results
    # The plain kernel keeps the residual parameters in its signature and never
    # reads them: bind them to ``x`` / ``y`` / 0 exactly like the source launcher.
    values: dict[str, object] = {
        "x": x,
        "w": w,
        "y": out,
        "r": rstd,
        "rows": rows,
        "eps": float(eps),
        "x_stride": int(x.stride(0)),
        "u": residual if has_residual else x,
        "h_new": residual_out if has_residual else out,
        "u_stride": int(residual.stride(0)) if has_residual else 0,
    }
    device_index = _device_index(device)
    for name in selected_modules(arch, hidden, "fwd", has_residual, rows=rows):
        _loader.run_module(name, values, rows=rows, device_index=device_index)
    _copy_back(copy_backs)
    return results


@flashinfer_api
def cake_rmsnorm_train_backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    *,
    deterministic: bool = True,
    g_residual: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fused BF16 RMSNorm backward: ``dx`` and the FP32 weight gradient ``dw``.

    Parameters
    ----------
    g : torch.Tensor
        BF16 ``[T, H]`` gradient of ``y`` of any layout.  Views with a contiguous
        feature dimension, 16-byte aligned storage and a row stride that is a
        multiple of 16 bytes (expanded or sliced autograd gradients included) are
        read in place; other layouts are materialized once.
    x : torch.Tensor
        The BF16 ``[T, H]`` tensor that was normalized (``x`` of the plain
        forward, ``h_new`` of the residual forward); same layout rules as ``g``.
    w : torch.Tensor
        BF16 ``[H]`` weight (copied when not contiguous).
    rstd : torch.Tensor
        FP32 ``[T]`` reciprocal RMS returned by the forward.
    deterministic : bool
        Only the deterministic fixed-order ``dw`` reduction is exported;
        ``False`` is rejected.
    g_residual : torch.Tensor, optional
        BF16 ``[T, H]`` gradient flowing directly through ``h_new`` (residual
        forward), any layout.  It is added once so that the returned ``dx`` is
        the gradient of both residual-add inputs.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_rmsnorm_train_backward_workspace`; allocated when omitted.
    dx, dw : torch.Tensor, optional
        Preallocated outputs: BF16 ``[T, H]`` and FP32 ``[H]``.  Contiguous
        aligned buffers are written directly; any other layout receives a copy
        after the launch.

    Returns
    -------
    ``(dx, dw)`` with ``dw`` in FP32; the caller decides whether to cast it.
    For a fixed token count and architecture ``dw`` is bitwise reproducible.
    """

    if not deterministic:
        raise ValueError(
            "cake_rmsnorm_train_backward exports only the deterministic dw reduction"
        )
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("x must be a [T, H] tensor")
    device = x.device
    rows, hidden = int(x.shape[0]), int(x.shape[1])
    arch = _device_arch(device)
    has_residual = g_residual is not None
    align = BACKWARD_ALIGNMENT_BYTES
    x = _kernel_input(
        x,
        "x",
        rows=rows,
        hidden=hidden,
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
    )
    w = _contiguous_input(
        w, "w", shape=(hidden,), device=device, dtype=torch.bfloat16, alignment=align
    )
    rstd = _contiguous_input(
        rstd, "rstd", shape=(rows,), device=device, dtype=torch.float32, alignment=4
    )
    g = _kernel_input(
        g,
        "g",
        rows=rows,
        hidden=hidden,
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
    )
    if has_residual:
        g_residual = _kernel_input(
            g_residual,
            "g_residual",
            rows=rows,
            hidden=hidden,
            device=device,
            dtype=torch.bfloat16,
            alignment=align,
        )
    if not route_applies(
        device_capability=tuple(torch.cuda.get_device_capability(device)),
        hidden=hidden,
        kind="bwd",
        residual=has_residual,
    ):
        raise ValueError(
            f"no exported Cake RMSNorm training backward for hidden={hidden} "
            f"(residual={has_residual}) on {arch}; supported hidden sizes: "
            f"{supported_hidden_sizes(arch)}"
        )
    dx, dx_user = _output_buffer(
        dx,
        "dx",
        shape=(rows, hidden),
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
    )
    dw, dw_user = _output_buffer(
        dw, "dw", shape=(hidden,), device=device, dtype=torch.float32, alignment=align
    )
    copy_backs = [(dx, dx_user), (dw, dw_user)]
    dx_result = dx_user if dx_user is not None else dx
    dw_result = dw_user if dw_user is not None else dw
    if rows == 0:
        dw_result.zero_()
        return dx_result, dw_result
    record = _workspace_record(arch, hidden, has_residual)
    if workspace is None:
        workspace = cake_rmsnorm_train_backward_workspace(
            rows, hidden, device, residual=has_residual
        )
    partial, counters, layout = _workspace_views(
        workspace, record, rows=rows, device=device
    )
    # The plain kernel keeps the residual-gradient parameters in its signature
    # and never reads them: bind them to ``g`` exactly like the source launcher.
    g_h = g_residual if has_residual else g
    values: dict[str, object] = {
        "g": g,
        "x": x,
        "w": w,
        "r": rstd,
        "dx": dx,
        "dw": dw,
        "partial": partial,
        "counters": counters,
        "g_h": g_h,
        "rows": rows,
        "g_stride": int(g.stride(0)),
        "x_stride": int(x.stride(0)),
        "g_h_stride": int(g_h.stride(0)),
        "n_chunks": int(layout["n_chunks"]),
        "rows_per_chunk": int(layout["rows_per_chunk"]),
    }
    device_index = _device_index(device)
    for name in selected_modules(arch, hidden, "bwd", has_residual, rows=rows):
        _loader.run_module(name, values, rows=rows, device_index=device_index)
    _copy_back(copy_backs)
    return dx_result, dw_result


class CakeRMSNormFunction(torch.autograd.Function):
    """Autograd wrapper saving only ``(x_or_h_new, w, rstd)`` for the backward.

    ``forward(x, w, eps, residual=None, deterministic=True)`` returns ``y`` or
    ``(y, h_new)``.  In residual mode the gradient of ``x`` and of ``residual``
    is the same tensor (``dx + g_h_new``).  The weight gradient is accumulated
    in FP32 by the kernel and returned in ``w.dtype``.
    """

    @staticmethod
    def forward(ctx, x, w, eps, residual=None, deterministic=True):
        outputs = cake_rmsnorm_train_forward(x, w, eps, residual=residual)
        y, rstd = outputs[0], outputs[1]
        normalized = outputs[2] if residual is not None else x
        ctx.save_for_backward(normalized, w, rstd)
        ctx.deterministic = bool(deterministic)
        ctx.has_residual = residual is not None
        if residual is not None:
            return y, outputs[2]
        return y

    @staticmethod
    def backward(ctx, grad_y, *grad_rest):
        normalized, w, rstd = ctx.saved_tensors
        grad_h = None
        if ctx.has_residual and grad_rest and grad_rest[0] is not None:
            grad_h = grad_rest[0].to(torch.bfloat16)
        if grad_y is None:
            grad_y = torch.zeros_like(normalized)
        dx, dw = cake_rmsnorm_train_backward(
            grad_y.to(torch.bfloat16),
            normalized,
            w,
            rstd,
            deterministic=ctx.deterministic,
            g_residual=grad_h,
        )
        dw = dw if w.dtype == torch.float32 else dw.to(w.dtype)
        return dx, dw, None, (dx if ctx.has_residual else None), None


@flashinfer_api
def cake_rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    deterministic: bool = True,
):
    """Autograd-enabled fused BF16 RMSNorm (optionally with a fused residual add).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, H]`` input on a supported CUDA device.
    w : torch.Tensor
        BF16 ``[H]`` RMSNorm weight.
    eps : float
        Variance epsilon evaluated in FP32 by the kernel.
    residual : torch.Tensor, optional
        BF16 ``[T, H]`` update added to ``x`` before normalization.  When
        provided, the normalized input ``h_new`` is returned with the output.
    deterministic : bool
        Whether to use the deterministic fixed-order weight-gradient reduction.
        Only ``True`` is supported; ``False`` is rejected during backward.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        BF16 ``y`` with shape ``[T, H]``, or ``(y, h_new)`` when ``residual``
        is given.  Backward launches the fused ``dx`` and deterministic ``dw``
        kernels of :func:`cake_rmsnorm_train_backward`.
    """

    return CakeRMSNormFunction.apply(x, w, eps, residual, deterministic)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormFunction",
    "cake_rmsnorm",
    "cake_rmsnorm_train_backward",
    "cake_rmsnorm_train_backward_workspace",
    "cake_rmsnorm_train_backward_workspace_bytes",
    "cake_rmsnorm_train_forward",
    "is_cake_rmsnorm_train_supported",
    "route_key",
]
