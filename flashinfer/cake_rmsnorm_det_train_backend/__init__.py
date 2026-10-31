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

Backend of :mod:`flashinfer.cake_rmsnorm_det_train`: Cake fused BF16 RMSNorm training
forward / backward (SM100 / SM103 / SM107) as one self-contained package (host
API bodies, the ``cake_jit`` loader and the generated CUDA sources under ``csrc/``).
The stable, ``@flashinfer_api``-decorated entry points and the full user-facing
documentation live in :mod:`flashinfer.cake_rmsnorm_det_train`; import that module.
"""

from __future__ import annotations

import math
import textwrap
from typing import Optional

import torch

from . import cake_jit as _loader
from . import cake_windows as _windows
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    BACKWARD_ALIGNMENT_BYTES,
    FORWARD_ALIGNMENT_BYTES,
    FP32_WEIGHT_ALIGNMENT_BYTES,
    arch_for_capability,
    program_windows,
    route_applies,
    route_key,
    route_modules,
    selected_modules,
    supported_hidden_sizes,
    weight_dtypes,
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


def is_cake_rmsnorm_det_train_supported(
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


def _row_view(tensor: torch.Tensor, name: str) -> tuple[torch.Tensor, tuple[int, ...]]:
    """``(the [rows, H] view of an input, its leading shape)``.

    A ``[T, H]`` tensor is its own view.  A tensor of more dimensions,
    ``[..., H]``, collapses its leading dimensions into rows when its feature
    dimension is contiguous (``stride(-1) == 1``) and the leading dimensions
    are contiguous with one another in row-major order --
    ``stride(i) == stride(i + 1) * size(i + 1)`` for every pair of leading
    dimensions of size other than one -- so that consecutive rows sit one
    stride apart (a ``[T, heads, H]`` slice of a wider ``[T, heads, W]``
    buffer, strides ``(heads * W, W, 1)``, qualifies).  The view keeps that
    row stride, so the alignment rules of the kernel inputs apply to it
    unchanged.  Any other layout raises ``ValueError`` naming the rule: the
    kernels index rows by one stride and write contiguous outputs.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} must be a [T, H] tensor (or [..., H] with more leading "
            f"dimensions), got shape {tuple(tensor.shape)}"
        )
    if tensor.ndim == 2:
        return tensor, (int(tensor.shape[0]),)
    lead = tuple(int(size) for size in tensor.shape[:-1])
    hidden = int(tensor.shape[-1])
    rows = math.prod(lead)
    if rows == 0 or hidden == 0:
        return tensor.reshape(rows, hidden), lead
    dims = [i for i in range(tensor.ndim - 1) if tensor.shape[i] != 1]
    collapsible = (tensor.stride(-1) == 1 or hidden == 1) and all(
        tensor.stride(outer) == tensor.stride(inner) * tensor.shape[inner]
        for outer, inner in zip(dims, dims[1:], strict=False)
    )
    if not collapsible:
        raise ValueError(
            f"{name} of shape {tuple(tensor.shape)} with strides "
            f"{tuple(tensor.stride())} cannot be viewed as [{rows}, {hidden}] "
            "rows: the feature dimension must be contiguous (stride(-1) == 1) "
            "and the leading dimensions contiguous with one another in "
            "row-major order (stride(i) == stride(i + 1) * size(i + 1)); "
            "reshape or copy the tensor into such a layout"
        )
    pitch = tensor.stride(dims[-1]) if dims else hidden
    view = tensor.as_strided((rows, hidden), (pitch, 1), tensor.storage_offset())
    return view, lead


def _companion_view(
    tensor: torch.Tensor, name: str, *, lead: tuple[int, ...], hidden: int
) -> torch.Tensor:
    """The ``[rows, H]`` view of an input that takes the first input's shape
    (``residual``; the backward's ``g``, ``x`` companions and ``g_residual``)."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected = (*lead, hidden)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
        )
    return _row_view(tensor, name)[0]


def _rows_view(
    tensor: torch.Tensor, *, lead: tuple[int, ...], rows: int
) -> torch.Tensor:
    """``rstd`` as ``[rows]``: the leading shape of an N-D call reshaped, anything else as is."""

    if (
        isinstance(tensor, torch.Tensor)
        and len(lead) > 1
        and tuple(tensor.shape) == tuple(lead)
    ):
        return tensor.reshape(rows)
    return tensor


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


# The weight dtypes the host accepts and their route tokens (``cake_jit.WEIGHT_DTYPES``).
_WEIGHT_DTYPES = {torch.bfloat16: "bf16", torch.float32: "f32"}


def _weight(
    w: torch.Tensor, *, hidden: int, device: torch.device, alignment: int
) -> tuple[torch.Tensor, str]:
    """``(the weight the kernel reads, its route token)``: a BF16 weight in place
    when contiguous and ``alignment``-byte aligned, an FP32 weight in place when
    contiguous and ``FP32_WEIGHT_ALIGNMENT_BYTES`` aligned (the FP32-weight
    programs load it with 32-byte vectors), else a copy; another dtype is refused."""

    if not isinstance(w, torch.Tensor):
        raise TypeError("w must be a torch.Tensor")
    token = _WEIGHT_DTYPES.get(w.dtype)
    if token is None:
        raise ValueError(f"w must be one of {tuple(_WEIGHT_DTYPES)}, got {w.dtype}")
    if token == "f32":
        alignment = FP32_WEIGHT_ALIGNMENT_BYTES
    kernel_w = _contiguous_input(
        w, "w", shape=(hidden,), device=device, dtype=w.dtype, alignment=alignment
    )
    return kernel_w, token


def _route_error(
    kind: str, hidden: int, residual: bool, w: torch.Tensor, arch: str
) -> ValueError:
    """The error of a call no exported route serves: the hidden sizes with routes
    on ``arch`` and the weight dtypes this route ships for."""

    word = "forward" if kind == "fwd" else "backward"
    shipped = weight_dtypes(arch, hidden, kind, residual)
    dtypes = tuple(dtype for dtype, token in _WEIGHT_DTYPES.items() if token in shipped)
    return ValueError(
        f"no exported Cake RMSNorm training {word} for hidden={hidden} "
        f"(residual={residual}, w={w.dtype}) on {arch}; supported hidden sizes: "
        f"{supported_hidden_sizes(arch)}; weight dtypes exported for this route: "
        f"{dtypes if dtypes else 'none'}"
    )


def _output_buffer(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    alignment: int,
    lead: Optional[tuple[int, ...]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``(kernel buffer, the tensor returned to the caller, user buffer to copy back into or None)``.

    A caller-provided output of any layout is honoured: the kernel writes a
    contiguous aligned buffer directly when the caller's tensor is one, otherwise
    into a temporary that is copied back into the caller's view after the launch.
    ``lead`` (an N-D call) is the leading shape the caller's buffer and the
    returned tensor have in place of the ``shape[0]`` rows of the kernel buffer.
    """

    user_shape = shape if lead is None else (*lead, *shape[1:])
    if tensor is None:
        kernel = torch.empty(shape, dtype=dtype, device=device)
        return kernel, (kernel if lead is None else kernel.view(user_shape)), None
    _validate_tensor(tensor, name, shape=user_shape, device=device, dtype=dtype)
    if tensor.numel() == 0 or (
        tensor.is_contiguous() and tensor.data_ptr() % alignment == 0
    ):
        return (tensor if lead is None else tensor.reshape(shape)), tensor, None
    return torch.empty(shape, dtype=dtype, device=device), tensor, tensor


def _copy_back(pairs: list[tuple[torch.Tensor, Optional[torch.Tensor]]]) -> None:
    for kernel_buffer, user_buffer in pairs:
        if user_buffer is not None:
            user_buffer.copy_(kernel_buffer.reshape(user_buffer.shape))


def _backward_module_records(
    arch: str, hidden: int, residual: bool, w_dtype: str = "bf16"
) -> list[dict]:
    return [
        _loader.MODULES[name]
        for name in route_modules(arch, hidden, "bwd", residual, w_dtype)
    ]


def _workspace_record(
    arch: str, hidden: int, residual: bool, rows: int, w_dtype: str = "bf16"
) -> dict:
    """The backward module that sizes the partial / counter workspace for ``rows`` tokens.

    Every main backward module of a route (the record owning a
    ``workspace_rule``; the dw reduce records own none) claims its chunks on
    ``slots = ctas_per_sm x SMs`` persistent CTAs; a route that switches
    modules on the row count is sized by the module with the largest partial
    capacity, whose layout every module of the route shares.
    """

    owners = [
        record
        for record in _backward_module_records(arch, hidden, residual, w_dtype)
        if record.get("workspace_rule") is not None
    ]
    if not owners:
        raise RuntimeError(
            f"the exported backward route for hidden={hidden} declares "
            "no workspace owner; expected at least one"
        )
    return max(
        owners,
        key=lambda record: (
            _loader.partial_capacity(record, rows, device_index=0),
            int(record["grid_rule"]["ctas_per_sm"]),
        ),
    )


def cake_rmsnorm_det_train_backward_workspace_bytes(
    rows: int, hidden: int, device: torch.device, *, residual: bool = False
) -> int:
    """Bytes of the caller-owned backward workspace of width ``hidden`` on ``device`` for up to ``rows`` tokens.

    The workspace holds the ``uint32`` chunk-claim and completion counters in
    its first ``cake_jit.COUNTERS_BYTES`` bytes, then the FP32 per-chunk ``dw``
    partial rows: one row per chunk of the backward chunk table, whose count
    is bounded by ``min(rows, MAX_ROUNDS x slots)`` of the route's largest
    persistent grid (``cake_jit.chunk_capacity``).  The size grows with
    ``rows`` and a buffer sized for ``rows`` serves every call with at most
    that many tokens.  The counters must be zero on first use (allocate with
    ``torch.zeros`` or use :func:`cake_rmsnorm_det_train_backward_workspace`); the
    kernels leave them zero, so the same buffer is reusable across calls and
    CUDA Graph replays without host writes.  The buffer serves the BF16-weight
    and the FP32-weight route of the width alike.
    """

    device = torch.device(device)
    rows = int(rows)
    hidden = int(hidden)
    if rows < 0:
        raise ValueError("rows must be non-negative")
    arch = _device_arch(device)
    device_index = _device_index(device)
    # Every module of a route shares its owner's partial capacity; the buffer
    # also holds the widest counter table any module of the routes declares.
    total = 0
    for w_dtype in weight_dtypes(arch, hidden, "bwd", residual) or ("bf16",):
        owner = _workspace_record(arch, hidden, residual, max(rows, 1), w_dtype)
        for record in _backward_module_records(arch, hidden, residual, w_dtype):
            if record.get("workspace_rule") is None:
                continue
            layout = _loader.workspace_layout(
                record, max(rows, 1), device_index=device_index, capacity_record=owner
            )
            total = max(total, int(layout["total_bytes"]))
    return total


def cake_rmsnorm_det_train_backward_workspace(
    rows: int, hidden: int, device: torch.device, *, residual: bool = False
) -> torch.Tensor:
    """Allocate a zero-initialized backward workspace (see ``..._workspace_bytes``)."""

    nbytes = cake_rmsnorm_det_train_backward_workspace_bytes(
        rows, hidden, device, residual=residual
    )
    return torch.zeros(nbytes, dtype=torch.uint8, device=torch.device(device))


def _workspace_views(
    workspace: torch.Tensor,
    record: dict,
    *,
    rows: int,
    device: torch.device,
    owner: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    layout = _loader.workspace_layout(
        record,
        rows,
        device_index=_device_index(device),
        capacity_record=record if owner is None else owner,
    )
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
            f"{layout['total_bytes']} (cake_rmsnorm_det_train_backward_workspace_bytes)"
        )
    hidden = int(record["hidden"])
    begin = layout["partial_offset"]
    partial = workspace[begin : begin + 4 * layout["n_chunks"] * hidden].view(
        torch.float32
    )
    partial = partial.view(layout["n_chunks"], hidden)
    begin = layout["counters_offset"]
    counters = workspace[begin : begin + 4 * max(layout["counters"], 1)].view(
        torch.uint32
    )
    return partial, counters, layout


def cake_rmsnorm_det_train_forward(
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
        odd row stride, transposed storage) are materialized once.  A tensor of
        more dimensions, ``[..., H]``, is normalized row by row when its feature
        dimension is contiguous and its leading dimensions are contiguous with
        one another in row-major order (``stride(i) == stride(i + 1) * size(i +
        1)``; a ``[T, heads, H]`` slice of a wider ``[T, heads, W]`` buffer
        qualifies): the leading dimensions collapse into ``prod(...)`` rows, the
        outputs take the input's leading shape (``rstd`` is ``[...]``), ``w`` is
        shared by every row and the backward's ``dw`` sums them all; any other
        N-D layout raises ``ValueError`` naming the rule.  With ``residual`` this
        is the residual stream ``h`` and the normalized value is
        ``h_new = BF16(h + residual)``.
    w : torch.Tensor
        BF16 or FP32 ``[H]`` weight.  BF16 is read in place when contiguous and
        32-byte aligned (the forward kernels load it with 32-byte vectors), FP32
        -- the effective weight ``1 + w`` a zero-centred model computes in FP32
        -- when contiguous and 32-byte aligned; any other layout is copied once.
        An FP32 weight needs the FP32-weight route of the width on the device
        (:func:`weight_dtypes`), else ``ValueError``.
    eps : float
        Variance epsilon (FP32 inside the kernel).
    residual : torch.Tensor, optional
        BF16 update of ``x``'s shape added to ``x`` before normalization (same
        layout rules as ``x``).  Enables the fused residual-add route.
    out, rstd, residual_out : torch.Tensor, optional
        Preallocated outputs: BF16 ``y`` of ``x``'s shape, FP32 reciprocal RMS of
        its leading shape (``[T]``) and (residual only) BF16 ``h_new`` of ``x``'s
        shape.  Contiguous aligned buffers are written directly; any other layout
        receives a copy after the launch.

    Returns
    -------
    ``(y, rstd)`` or ``(y, rstd, h_new)`` when ``residual`` is given.
    ``rstd`` is the FP32 per-row ``1 / sqrt(mean(h_new^2) + eps)`` the backward
    needs; no other ``[T, H]`` temporary is produced.
    """

    x, lead = _row_view(x, "x")
    nd = lead if len(lead) > 1 else None
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
    w, w_dtype = _weight(w, hidden=hidden, device=device, alignment=align)
    if has_residual:
        residual = _kernel_input(
            _companion_view(residual, "residual", lead=lead, hidden=hidden),
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
        w_dtype=w_dtype,
    ):
        raise _route_error("fwd", hidden, has_residual, w, arch)
    out, y_result, out_user = _output_buffer(
        out,
        "out",
        shape=(rows, hidden),
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
        lead=nd,
    )
    rstd, rstd_result, rstd_user = _output_buffer(
        rstd,
        "rstd",
        shape=(rows,),
        device=device,
        dtype=torch.float32,
        alignment=BACKWARD_ALIGNMENT_BYTES,
        lead=nd,
    )
    residual_out_user = None
    results: tuple[torch.Tensor, ...] = (y_result, rstd_result)
    if has_residual:
        residual_out, h_new_result, residual_out_user = _output_buffer(
            residual_out,
            "residual_out",
            shape=(rows, hidden),
            device=device,
            dtype=torch.bfloat16,
            alignment=align,
            lead=nd,
        )
        results = (y_result, rstd_result, h_new_result)
    copy_backs = [(out, out_user), (rstd, rstd_user), (residual_out, residual_out_user)]
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
    for name in selected_modules(
        arch, hidden, "fwd", has_residual, rows=rows, w_dtype=w_dtype
    ):
        _loader.run_module(name, values, rows=rows, device_index=device_index)
    _copy_back(copy_backs)
    return results


def cake_rmsnorm_det_train_backward(
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
        read in place; other layouts are materialized once.  An N-D gradient,
        ``[..., H]``, takes ``x``'s shape and collapses its leading dimensions
        into rows under the forward's rule.
    x : torch.Tensor
        The BF16 ``[T, H]`` (or ``[..., H]``) tensor that was normalized (``x`` of
        the plain forward, ``h_new`` of the residual forward); same layout rules
        as the forward's ``x``.
    w : torch.Tensor
        BF16 or FP32 ``[H]`` weight (the forward's).  BF16 is read in place when
        contiguous and 16-byte aligned, FP32 when contiguous and 32-byte aligned;
        any other layout is copied once.  An FP32 weight needs the FP32-weight
        route of the width (:func:`weight_dtypes`), else ``ValueError``.
    rstd : torch.Tensor
        FP32 ``[T]`` reciprocal RMS returned by the forward (of the leading
        shape ``[...]`` for an N-D call, or flat ``[prod(...)]``).
    deterministic : bool
        Only the deterministic fixed-order ``dw`` reduction is exported;
        ``False`` is rejected.
    g_residual : torch.Tensor, optional
        BF16 gradient of ``g``'s shape flowing directly through ``h_new``
        (residual forward), any layout.  It is added once so that the returned
        ``dx`` is the gradient of both residual-add inputs.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_rmsnorm_det_train_backward_workspace` sized for at least this
        many rows; allocated when omitted.
    dx, dw : torch.Tensor, optional
        Preallocated outputs: BF16 ``dx`` of ``x``'s shape and FP32 ``[H]``.
        Contiguous aligned buffers are written directly; any other layout
        receives a copy after the launch.

    Returns
    -------
    ``(dx, dw)`` with ``dw`` in FP32; the caller decides whether to cast it.
    For a fixed token count and architecture ``dw`` is bitwise reproducible.
    """

    if not deterministic:
        raise ValueError(
            "cake_rmsnorm_det_train_backward exports only the deterministic dw reduction"
        )
    x, lead = _row_view(x, "x")
    nd = lead if len(lead) > 1 else None
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
    w, w_dtype = _weight(w, hidden=hidden, device=device, alignment=align)
    rstd = _contiguous_input(
        _rows_view(rstd, lead=lead, rows=rows),
        "rstd",
        shape=(rows,),
        device=device,
        dtype=torch.float32,
        alignment=4,
    )
    g = _kernel_input(
        _companion_view(g, "g", lead=lead, hidden=hidden),
        "g",
        rows=rows,
        hidden=hidden,
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
    )
    if has_residual:
        g_residual = _kernel_input(
            _companion_view(g_residual, "g_residual", lead=lead, hidden=hidden),
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
        w_dtype=w_dtype,
    ):
        raise _route_error("bwd", hidden, has_residual, w, arch)
    dx, dx_result, dx_user = _output_buffer(
        dx,
        "dx",
        shape=(rows, hidden),
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
        lead=nd,
    )
    dw, dw_result, dw_user = _output_buffer(
        dw, "dw", shape=(hidden,), device=device, dtype=torch.float32, alignment=align
    )
    copy_backs = [(dx, dx_user), (dw, dw_user)]
    if rows == 0:
        dw_result.zero_()
        return dx_result, dw_result
    owner = _workspace_record(arch, hidden, has_residual, rows, w_dtype)
    names = selected_modules(
        arch, hidden, "bwd", has_residual, rows=rows, w_dtype=w_dtype
    )
    launched = [
        _loader.MODULES[name]
        for name in names
        if _loader.MODULES[name].get("workspace_rule") is not None
    ]
    if len(launched) != 1:
        raise RuntimeError(
            f"the exported backward route for hidden={hidden} launches "
            f"{len(launched)} workspace-owning modules for {rows} rows; expected one"
        )
    if workspace is None:
        workspace = cake_rmsnorm_det_train_backward_workspace(
            rows, hidden, device, residual=has_residual
        )
    # The chunk table is the launched main module's (its slot count and rows
    # per pipeline step); the partial capacity is the route's.  The main kernel
    # claims the chunks from the counters and writes one partial row per chunk;
    # the dw reduce record launched after it folds the ``n_chunks`` rows.
    partial, counters, layout = _workspace_views(
        workspace, launched[0], rows=rows, device=device, owner=owner
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
        "slots": int(layout["slots"]),
    }
    device_index = _device_index(device)
    for name in names:
        _loader.run_module(name, values, rows=rows, device_index=device_index)
    _copy_back(copy_backs)
    return dx_result, dw_result


class CakeRMSNormDetFunction(torch.autograd.Function):
    """Autograd wrapper saving only ``(x_or_h_new, w, rstd)`` for the backward.

    ``forward(x, w, eps, residual=None, deterministic=True)`` returns ``y`` or
    ``(y, h_new)``.  In residual mode the gradient of ``x`` and of ``residual``
    is the same tensor (``dx + g_h_new``).  The weight gradient is accumulated
    in FP32 by the kernel and returned in ``w.dtype``.
    """

    @staticmethod
    def forward(ctx, x, w, eps, residual=None, deterministic=True):
        outputs = cake_rmsnorm_det_train_forward(x, w, eps, residual=residual)
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
        dx, dw = cake_rmsnorm_det_train_backward(
            grad_y.to(torch.bfloat16),
            normalized,
            w,
            rstd,
            deterministic=ctx.deterministic,
            g_residual=grad_h,
        )
        dw = dw if w.dtype == torch.float32 else dw.to(w.dtype)
        return dx, dw, None, (dx if ctx.has_residual else None), None


def cake_rmsnorm_det(
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
        BF16 ``[T, H]`` input on a supported CUDA device, or ``[..., H]`` under
        the layout rule of :func:`cake_rmsnorm_det_train_forward` (its leading
        dimensions collapse into rows).
    w : torch.Tensor
        BF16 or FP32 ``[H]`` RMSNorm weight (FP32 on the routes that ship the
        FP32-weight programs, see :func:`weight_dtypes`); the weight gradient is
        returned in ``w.dtype``.
    eps : float
        Variance epsilon evaluated in FP32 by the kernel.
    residual : torch.Tensor, optional
        BF16 update of ``x``'s shape added to ``x`` before normalization.  When
        provided, the normalized input ``h_new`` is returned with the output.
    deterministic : bool
        Whether to use the deterministic fixed-order weight-gradient reduction.
        Only ``True`` is supported; ``False`` is rejected during backward.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        BF16 ``y`` of ``x``'s shape, or ``(y, h_new)`` when ``residual``
        is given.  Backward launches the fused ``dx`` and deterministic ``dw``
        kernels of :func:`cake_rmsnorm_det_train_backward`.
    """

    return CakeRMSNormDetFunction.apply(x, w, eps, residual, deterministic)


def _with_program_selection(doc: Optional[str], kind: str) -> str:
    """``doc`` with the "Program selection by token count" notes of the routes of ``kind`` appended.

    The notes are the exporter's rendering of the registry (``cake_windows``):
    the row-count windows of every route, the measured outcome at every window
    edge and the contract of the outputs across the windows, so the documented
    thresholds are the registry's.  ``program_windows`` returns the windows.
    """

    section = textwrap.indent(_windows.SECTIONS[kind].strip("\n"), "    ")
    return f"{(doc or '').rstrip()}\n\n    Notes\n    -----\n{section}\n    "


cake_rmsnorm_det_train_forward.__doc__ = _with_program_selection(
    cake_rmsnorm_det_train_forward.__doc__, "fwd"
)
cake_rmsnorm_det_train_backward.__doc__ = _with_program_selection(
    cake_rmsnorm_det_train_backward.__doc__, "bwd"
)


# The names before the det rename, the same objects as the det names of this backend package (not in __all__).
CakeRMSNormFunction = CakeRMSNormDetFunction
cake_rmsnorm = cake_rmsnorm_det
cake_rmsnorm_train_backward = cake_rmsnorm_det_train_backward
cake_rmsnorm_train_backward_workspace = cake_rmsnorm_det_train_backward_workspace
cake_rmsnorm_train_backward_workspace_bytes = (
    cake_rmsnorm_det_train_backward_workspace_bytes
)
cake_rmsnorm_train_forward = cake_rmsnorm_det_train_forward
is_cake_rmsnorm_train_supported = is_cake_rmsnorm_det_train_supported


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormDetFunction",
    "cake_rmsnorm_det",
    "cake_rmsnorm_det_train_backward",
    "cake_rmsnorm_det_train_backward_workspace",
    "cake_rmsnorm_det_train_backward_workspace_bytes",
    "cake_rmsnorm_det_train_forward",
    "is_cake_rmsnorm_det_train_supported",
    "program_windows",
    "route_key",
    "weight_dtypes",
]
