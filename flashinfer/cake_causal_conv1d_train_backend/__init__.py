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

Backend of :mod:`flashinfer.cake_causal_conv1d_train`: the Cake packed width-4
depthwise causal convolution + SiLU training forward / backward (SM100 / SM103 /
SM107) as one self-contained package (host API bodies, the ``cake_jit`` registry,
the pinned ``cake_build`` loader and the generated CUDA sources under ``csrc/``).
The stable, ``@flashinfer_api``-decorated entry points and the full user-facing
documentation live in :mod:`flashinfer.cake_causal_conv1d_train`; import that
module.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    CHANNEL_MULTIPLE,
    INT32_MAX,
    WIDTH,
    arch_for_capability,
    launch_grid,
    partial_bytes,
    partial_rows,
    plan,
    route_applies,
    route_modules,
    run_module,
    selected_modules,
    workspace_layout,
    workspace_owners,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _arch_of_index(device_index: int) -> Optional[str]:
    """The exported architecture of one CUDA device index, queried once per process."""

    return arch_for_capability(tuple(torch.cuda.get_device_capability(device_index)))


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(
            f"Cake causal conv1d training requires CUDA tensors, got {device}"
        )
    arch = _arch_of_index(_device_index(device))
    if arch is None:
        raise ValueError(
            "Cake causal conv1d training supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got "
            f"{tuple(torch.cuda.get_device_capability(device))}"
        )
    return arch


def is_cake_causal_conv1d_train_supported(device: torch.device) -> bool:
    """Whether ``device`` has an exported forward and backward route."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    capability = tuple(torch.cuda.get_device_capability(device))
    return all(
        route_applies(device_capability=capability, kind=kind)
        for kind in ("fwd", "bwd")
    )


def _require_tensor(tensor: object, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    return tensor


def _check_placement(
    tensor: torch.Tensor, name: str, *, device: torch.device, dtype: torch.dtype
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _tokens(
    tensor: object,
    name: str,
    *,
    device: torch.device,
    shape: Optional[tuple[int, ...]] = None,
) -> tuple[int, int, torch.Tensor]:
    """Validate a BF16 ``[S, C]`` or ``[1, S, C]`` token tensor; return ``(S, C, [S, C] view)``.

    ``shape`` pins the exact shape another token tensor of the call has (``dy``
    against ``x``).  Both kernels read the packed rows as 16-byte vectors of eight
    channels, so ``C`` is a multiple of eight; ``S`` and ``C`` are 32-bit kernel
    scalars (``S x C`` itself may exceed 2^31: the kernels address with 64-bit
    offsets).
    """

    t = _require_tensor(tensor, name)
    if t.ndim == 3 and t.shape[0] == 1:
        S, C = int(t.shape[1]), int(t.shape[2])
    elif t.ndim == 2:
        S, C = int(t.shape[0]), int(t.shape[1])
    else:
        raise ValueError(
            f"{name} must be a [S, C] or [1, S, C] tensor, got shape {tuple(t.shape)}"
        )
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(t.shape)}")
    _check_placement(t, name, device=device, dtype=torch.bfloat16)
    if C <= 0 or C % CHANNEL_MULTIPLE:
        raise ValueError(
            f"{name} must have a channel count C that is a positive multiple of "
            f"{CHANNEL_MULTIPLE}, got C={C}"
        )
    if S > INT32_MAX or C > INT32_MAX:
        raise ValueError(
            f"{name}: S={S} and C={C} must each fit a 32-bit kernel scalar"
        )
    return S, C, t.view(S, C)


def _weight(
    weight: object, C: int, *, device: torch.device, name: str = "weight"
) -> torch.Tensor:
    w = _require_tensor(weight, name)
    if w.ndim != 2 or int(w.shape[1]) != WIDTH:
        raise ValueError(
            f"{name} must be a [C, {WIDTH}] tensor of a width-{WIDTH} convolution "
            f"(tap {WIDTH - 1} multiplies the current token), got shape {tuple(w.shape)}"
        )
    if int(w.shape[0]) != C:
        raise ValueError(
            f"{name} must have one row per channel ([{C}, {WIDTH}]), got shape {tuple(w.shape)}"
        )
    _check_placement(w, name, device=device, dtype=torch.float32)
    return w


def _cu_seqlens(
    cu_seqlens: object, *, device: torch.device
) -> tuple[torch.Tensor, int]:
    """Validate the ``int32 [num_docs + 1]`` device prefix sums; return ``(cu_seqlens, num_docs)``.

    The host reads the tensor's metadata only: no element is copied to the host
    and no stream is synchronized, so the call is CUDA Graph capturable.
    """

    cu = _require_tensor(cu_seqlens, "cu_seqlens")
    if cu.ndim != 1:
        raise ValueError(
            f"cu_seqlens must be a 1-D [num_docs + 1] tensor, got shape {tuple(cu.shape)}"
        )
    _check_placement(cu, "cu_seqlens", device=device, dtype=torch.int32)
    num_docs = int(cu.numel()) - 1
    if num_docs < 1:
        raise ValueError(
            "cu_seqlens must hold at least two prefix sums (one document), got "
            f"{int(cu.numel())} elements"
        )
    return cu, num_docs


def _output(
    tensor: Optional[object],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """A caller-provided output (exact shape and dtype, contiguous) or a fresh one."""

    if tensor is None:
        return torch.empty(shape, dtype=dtype, device=device)
    out = _require_tensor(tensor, name)
    if tuple(out.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(out.shape)}"
        )
    _check_placement(out, name, device=device, dtype=dtype)
    return out


def _scalars(scalars: dict[str, int]) -> dict[str, int]:
    return {key: int(scalars[key]) for key in _loader.LOGICAL_SCALARS}


def _launch(
    record: dict, S: int, C: int, num_docs: int, *, device_index: int
) -> tuple[dict[str, int], int]:
    """The kernel scalars and the grid of one launch of ``record`` (the host plan of ``cake_jit``).

    CTA ``b`` owns channel tile ``b % n_ch_tiles``, so every channel tile needs
    a CTA of the launched program's grid on the device: a ``C`` beyond ``grid
    x threads x vec`` channels is refused rather than left partly unwritten.
    """

    scalars = plan(S, C, num_docs, geometry=record["grid_rule"]["geometry"])
    grid_x = launch_grid(record, device_index=device_index)[0]
    if scalars["n_ch_tiles"] > grid_x:
        raise ValueError(
            f"C={C} spans {scalars['n_ch_tiles']} channel tiles but the launch grid on this "
            f"device is {grid_x} CTAs; every channel tile needs a CTA"
        )
    return scalars, grid_x


def cake_causal_conv1d_train_backward_workspace_bytes(
    S: int, C: int, device: torch.device
) -> int:
    """Bytes of the caller-owned backward workspace for ``C`` channels on ``device``.

    The workspace holds the FP32 ``[rows, C, 4]`` ``dweight`` partials of the
    cooperative backward, one row per ``n_ch_tiles`` CTAs of a program's
    launch grid (``rows = ceil(grid / n_ch_tiles)``), sized by the largest
    grid of the architecture's backward programs: the channel count selects
    the program a call runs, and every program writes and reduces the leading
    rows of its own grid inside this buffer.  It depends on ``C`` and the
    device only, so it is independent of ``S`` and one buffer serves every
    token count; the kernel fully writes the rows it reduces, so the buffer
    needs no zeroing between launches or CUDA Graph replays.
    """

    device = torch.device(device)
    S, C = int(S), int(C)
    if S < 0:
        raise ValueError("S must be non-negative")
    if C <= 0 or C % CHANNEL_MULTIPLE:
        raise ValueError(
            f"C must be a positive multiple of {CHANNEL_MULTIPLE}, got {C}"
        )
    arch = _device_arch(device)
    device_index = _device_index(device)
    return max(
        int(workspace_layout(owner, C, device_index=device_index)["total_bytes"])
        for owner in workspace_owners(arch)
    )


def cake_causal_conv1d_train_backward_workspace(
    S: int, C: int, device: torch.device
) -> torch.Tensor:
    """Allocate a backward workspace (see ``..._workspace_bytes``; no zeroing is needed)."""

    nbytes = cake_causal_conv1d_train_backward_workspace_bytes(S, C, device)
    return torch.empty(nbytes, dtype=torch.uint8, device=torch.device(device))


def _partial_view(
    workspace: object, record: dict, *, C: int, rows: int, device: torch.device
) -> torch.Tensor:
    """The FP32 ``[rows, C, taps]`` view of this launch's partial rows inside ``workspace``."""

    ws = _require_tensor(workspace, "workspace")
    if ws.ndim != 1:
        raise ValueError(
            f"workspace must be a 1-D uint8 CUDA tensor, got shape {tuple(ws.shape)}"
        )
    _check_placement(ws, "workspace", device=device, dtype=torch.uint8)
    if ws.data_ptr() % 16 != 0:
        raise ValueError("workspace must be 16-byte aligned")
    nbytes = partial_bytes(record, C, rows)
    if ws.numel() < nbytes:
        raise ValueError(
            f"workspace holds {ws.numel()} bytes; this call needs at least "
            f"{nbytes}; allocate "
            "cake_causal_conv1d_train_backward_workspace_bytes(S, C, device) bytes"
        )
    taps = int(record["workspace_rule"]["taps"])
    return ws[:nbytes].view(torch.float32).view(rows, C, taps)


def cake_causal_conv1d_train_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Packed width-4 depthwise causal convolution + SiLU, training forward.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations of every
        document; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the current
        token, ``weight[c, 0]`` the token three positions back.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` contiguous device prefix sums of the document
        lengths (``cu_seqlens[0] == 0``, ``cu_seqlens[-1] == S``; a repeated
        value is an empty document).  The host reads its metadata only.
    out : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape, contiguous.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape: ``y[t, c] = bf16(silu(sum_k weight[c, k] *
        x[t - 3 + k, c]))`` with zero history before each document's first
        token.  ``S == 0`` returns the empty output without launching.
    """

    x = _require_tensor(x, "x")
    device = x.device
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x", device=device)
    w = _weight(weight, C, device=device)
    cu, num_docs = _cu_seqlens(cu_seqlens, device=device)
    y = _output(out, "out", shape=tuple(x.shape), dtype=torch.bfloat16, device=device)
    if S == 0:
        return y
    names = route_modules(arch, "fwd")
    device_index = _device_index(device)
    scalars, _ = _launch(
        _loader.MODULES[names[0]], S, C, num_docs, device_index=device_index
    )
    values: dict[str, object] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "y": y.view(S, C),
        **_scalars(scalars),
    }
    for name in names:
        run_module(name, values, device_index=device_index)
    return y


def cake_causal_conv1d_train_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dy: torch.Tensor,
    *,
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward.

    Parameters
    ----------
    x, weight, cu_seqlens
        The forward's inputs (same contracts as
        :func:`cake_causal_conv1d_train_forward`); nothing else is saved.
    dy : torch.Tensor
        BF16 gradient of ``y``, ``x``'s shape, contiguous.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_causal_conv1d_train_backward_workspace`; allocated when
        omitted.
    dx, dweight : torch.Tensor, optional
        Preallocated outputs: BF16 of ``x``'s shape and FP32 ``[C, 4]``, both
        contiguous.

    Returns
    -------
    ``(dx, dweight)``: BF16 ``dx`` of ``x``'s shape and the FP32 ``[C, 4]``
    weight gradient, accumulated in a fixed order, so ``dweight`` is bitwise
    reproducible for a given ``(S, C, device)``.  ``S == 0`` returns the empty
    ``dx`` and a zero ``dweight`` without launching.

    The architecture's backward route may ship several programs of the
    kernel; a call runs exactly one, chosen by ``C`` through the registry's
    host-side rule (``cake_jit.selected_modules``; the data is never read),
    so a given ``(S, C, device)`` always runs the same program.  The
    ``workspace`` is sized for every program of the route
    (:func:`cake_causal_conv1d_train_backward_workspace_bytes`); the selected
    program uses the leading rows of its own grid.  Build every library of
    the device's architecture (``cake_jit.specs_for_arch``) before capturing
    a CUDA Graph: the capture holds the program selected for its shape, and
    no JIT build may happen inside it.
    """

    x = _require_tensor(x, "x")
    device = x.device
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x", device=device)
    w = _weight(weight, C, device=device)
    cu, num_docs = _cu_seqlens(cu_seqlens, device=device)
    _, _, dy_flat = _tokens(dy, "dy", device=device, shape=tuple(x.shape))
    dx_out = _output(
        dx, "dx", shape=tuple(x.shape), dtype=torch.bfloat16, device=device
    )
    dw_out = _output(
        dweight, "dweight", shape=(C, WIDTH), dtype=torch.float32, device=device
    )
    if S == 0:
        dw_out.zero_()
        return dx_out, dw_out
    # The program is a pure function of the architecture and C (S never enters):
    # its own grid sizes the partial rows this call writes inside the workspace.
    names = selected_modules(arch, "bwd", C=C, S=S)
    record = _loader.MODULES[names[0]]
    if record["workspace_rule"] is None:
        raise RuntimeError(
            f"the backward program {names[0]} selected for C={C} on {arch} owns no "
            "partial workspace"
        )
    device_index = _device_index(device)
    scalars, grid_x = _launch(record, S, C, num_docs, device_index=device_index)
    if workspace is None:
        workspace = cake_causal_conv1d_train_backward_workspace(S, C, device)
    rows = partial_rows(grid_x, scalars["n_ch_tiles"])
    partial = _partial_view(workspace, record, C=C, rows=rows, device=device)
    values: dict[str, object] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "dy": dy_flat,
        "dx": dx_out.view(S, C),
        "dw": dw_out,
        "partial": partial,
        **_scalars(scalars),
    }
    for name in names:
        run_module(name, values, device_index=device_index)
    return dx_out, dw_out


class CakeCausalConv1dFunction(torch.autograd.Function):
    """Autograd wrapper saving only ``(x, weight, cu_seqlens)`` for the backward.

    ``forward(x, weight, cu_seqlens)`` returns ``y``; the backward launches the
    fused ``dx`` + deterministic FP32 ``dweight`` kernel of
    :func:`cake_causal_conv1d_train_backward` on the incoming gradient
    (materialized as a contiguous BF16 tensor when autograd hands over another
    layout or dtype).
    """

    @staticmethod
    def forward(ctx, x, weight, cu_seqlens):
        y = cake_causal_conv1d_train_forward(x, weight, cu_seqlens)
        ctx.save_for_backward(x, weight, cu_seqlens)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, cu_seqlens = ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x)
        dy = grad_y.to(torch.bfloat16).contiguous()
        dx, dweight = cake_causal_conv1d_train_backward(x, weight, cu_seqlens, dy)
        return dx, dweight, None


def cake_causal_conv1d(
    x: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations on a
        supported CUDA device.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` device prefix sums of the document lengths.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.  The backward saves only ``x``, ``weight``
        and ``cu_seqlens`` and returns BF16 ``dx`` and the FP32 ``dweight``.
    """

    return CakeCausalConv1dFunction.apply(x, weight, cu_seqlens)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeCausalConv1dFunction",
    "cake_causal_conv1d",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes",
    "cake_causal_conv1d_train_forward",
    "is_cake_causal_conv1d_train_supported",
]
