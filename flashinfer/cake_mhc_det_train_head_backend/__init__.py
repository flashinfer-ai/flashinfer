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

Backend of the Cake mHC training head of :mod:`flashinfer.cake_mhc_det_train`: the
stream mean ``Y = mean_i X[:, i, :]`` of the four residual streams and its
gradient (hidden 4096, 5120 and 7168; SM100 / SM103 / SM107) as one
self-contained package (host API bodies, the ``cake_jit`` loader and the
generated CUDA sources under ``csrc/``).  The stable,
``@flashinfer_api``-decorated entry points and the user-facing documentation
live in :mod:`flashinfer.cake_mhc_det_train`; import that module.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    MAX_TOKENS,
    STREAMS,
    arch_for_capability,
    supported_hidden_sizes,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
# Every kernel tensor is read and written with 256-bit vector accesses:
# contiguous storage, 32-byte aligned base (a row of 4096, 5120 or 7168 BF16
# values is a whole number of 32-byte vectors).
TENSOR_ALIGNMENT_BYTES = 32
_BF16 = torch.bfloat16


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


# The exported architecture of each CUDA device, resolved once per device index.
_DEVICE_ARCH: dict[int, str] = {}


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(
            f"the Cake mHC training head requires CUDA tensors, got {device}"
        )
    index = _device_index(device)
    arch = _DEVICE_ARCH.get(index)
    if arch is None:
        capability = tuple(torch.cuda.get_device_capability(index))
        arch = arch_for_capability(capability)
        if arch is None:
            raise ValueError(
                f"the Cake mHC training head supports compute capability {sorted(ARCH_BY_CAPABILITY)} only, "
                f"got {capability}"
            )
        _DEVICE_ARCH[index] = arch
    return arch


def is_cake_mhc_train_head_supported(
    device: torch.device, hidden: Optional[int] = None
) -> bool:
    """Whether ``device`` (and optionally the hidden size ``hidden``) has the exported head programs."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    arch = arch_for_capability(tuple(torch.cuda.get_device_capability(device)))
    if arch is None:
        return False
    if hidden is None:
        return bool(supported_hidden_sizes(arch))
    return int(hidden) in supported_hidden_sizes(arch)


def _validate(
    tensor: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """A kernel operand: exact shape, BF16, the device, contiguous, 32-byte aligned storage.

    No layout is materialized silently: a view the kernels cannot read in place
    is a ``ValueError`` naming the argument (the caller decides whether to copy).
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != _BF16:
        raise ValueError(f"{name} must be {_BF16}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        return tensor
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous (strides {tuple(tensor.stride())} for shape {shape})"
        )
    if tensor.data_ptr() % TENSOR_ALIGNMENT_BYTES:
        raise ValueError(
            f"{name} must start at a {TENSOR_ALIGNMENT_BYTES}-byte aligned address"
        )
    return tensor


def _rows_hidden(
    tensor: Any, name: str, ndim: int
) -> tuple[int, int, torch.device, str]:
    """``(T, hidden, device, arch)`` of the call's input ``tensor`` (``[T, 4, hidden]`` or ``[T, hidden]``)."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    spelled = f"[T, {STREAMS}, hidden]" if ndim == 3 else "[T, hidden]"
    if tensor.ndim != ndim or (ndim == 3 and tensor.shape[1] != STREAMS):
        raise ValueError(
            f"{name} must be a {spelled} tensor, got {tuple(tensor.shape)}"
        )
    rows, hidden = int(tensor.shape[0]), int(tensor.shape[-1])
    arch = _device_arch(tensor.device)
    if hidden not in supported_hidden_sizes(arch):
        raise ValueError(
            f"the Cake mHC training head on {arch} serves hidden sizes {supported_hidden_sizes(arch)}, got {hidden}"
        )
    if rows > MAX_TOKENS:
        raise ValueError(
            f"the Cake mHC training head serves at most {MAX_TOKENS} tokens, got T = {rows}"
        )
    return rows, hidden, tensor.device, arch


def _disjoint(a: torch.Tensor, b: torch.Tensor, names: str) -> None:
    if a.numel() == 0 or b.numel() == 0:
        return
    a0, b0 = a.data_ptr(), b.data_ptr()
    a1, b1 = a0 + a.numel() * a.element_size(), b0 + b.numel() * b.element_size()
    if a0 < b1 and b0 < a1:
        raise ValueError(f"{names} must not overlap")


_TVM_FFI: Any = None
_TVM_DEVICES: dict[int, Any] = {}


def _tvm_ffi() -> Any:
    global _TVM_FFI
    if _TVM_FFI is None:
        import tvm_ffi

        _TVM_FFI = tvm_ffi
    return _TVM_FFI


def _tvm_device(device_index: int) -> Any:
    found = _TVM_DEVICES.get(device_index)
    if found is None:
        found = _TVM_DEVICES[device_index] = _tvm_ffi().device("cuda", device_index)
    return found


def _run(
    role: str,
    arch: str,
    values: dict[str, Any],
    *,
    rows: int,
    hidden: int,
    device: torch.device,
) -> None:
    """Launch the module of ``role`` on the current torch stream of ``device``."""

    name = _loader.ROLES[role]
    record = _loader.MODULES[name]
    grid_x, grid_y, grid_z = _loader.launch_grid(record, rows, hidden)
    grid = {"grid_x": grid_x, "grid_y": grid_y, "grid_z": grid_z}
    values = {**values, "rows": rows, "hv": _loader.vectors_per_token(record, hidden)}
    logical = record["logical"]
    args = [
        grid[key] if kind == "grid" else values[logical[key]]
        for kind, key in record["arg_plan"]
    ]
    run = getattr(_loader.load(name, arch), record["ffi_entry"])
    device_index = _device_index(device)
    stream = torch.cuda.current_stream(device_index).cuda_stream
    with _tvm_ffi().use_raw_stream(_tvm_device(device_index), stream):
        run(*args)


def cake_mhc_train_head_forward(
    X: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``Y = BF16(((X0 + X1) + X2 + X3) * 0.25)``, bit-identical to ``X.mean(dim=1)``; see
    :func:`flashinfer.cake_mhc_det_train.cake_mhc_train_head_forward`."""

    rows, hidden, device, arch = _rows_hidden(X, "X", 3)
    X = _validate(X, "X", shape=(rows, STREAMS, hidden), device=device)
    if out is None:
        Y = torch.empty((rows, hidden), dtype=_BF16, device=device)
    else:
        Y = _validate(out, "out", shape=(rows, hidden), device=device)
        _disjoint(X, Y, "X and out")
    if rows == 0:
        return Y
    _run("head_fwd", arch, {"x": X, "y": Y}, rows=rows, hidden=hidden, device=device)
    return Y


def cake_mhc_train_head_backward(
    dY: torch.Tensor,
    *,
    dX: Optional[torch.Tensor] = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """``dX[:, i, :] = BF16(dY * 0.25)`` for each stream, or ``dX += BF16(dY * 0.25)``; see
    :func:`flashinfer.cake_mhc_det_train.cake_mhc_train_head_backward`."""

    rows, hidden, device, arch = _rows_hidden(dY, "dY", 2)
    dY = _validate(dY, "dY", shape=(rows, hidden), device=device)
    if dX is None:
        if accumulate:
            raise ValueError("accumulate=True adds into a caller-provided dX; pass dX")
        dX = torch.empty((rows, STREAMS, hidden), dtype=_BF16, device=device)
    else:
        dX = _validate(dX, "dX", shape=(rows, STREAMS, hidden), device=device)
        _disjoint(dY, dX, "dY and dX")
    if rows == 0:
        return dX
    role = "head_bwd_acc" if accumulate else "head_bwd"
    _run(role, arch, {"dy": dY, "dx": dX}, rows=rows, hidden=hidden, device=device)
    return dX


class CakeMHCTrainHeadFunction(torch.autograd.Function):
    """Autograd node of the head: forward :func:`cake_mhc_train_head_forward`,
    backward :func:`cake_mhc_train_head_backward`.  The backward needs only the
    incoming gradient, so the node saves no tensor: an activation checkpoint
    around it recomputes nothing for its backward and keeps nothing alive."""

    @staticmethod
    def forward(ctx: Any, X: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.set_materialize_grads(False)
        return cake_mhc_train_head_forward(X)

    @staticmethod
    def backward(ctx: Any, dY: Optional[torch.Tensor]) -> Optional[torch.Tensor]:  # type: ignore[override]
        if dY is None:
            return None
        # The incoming gradient may be any view (an expanded zero-stride
        # gradient, an offset slice): the kernel reads a contiguous, aligned copy.
        dY = dY.contiguous()
        if dY.data_ptr() % TENSOR_ALIGNMENT_BYTES:
            dY = dY.clone()
        return cake_mhc_train_head_backward(dY)


def cake_mhc_train_head(X: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled :func:`cake_mhc_train_head_forward`; see
    :func:`flashinfer.cake_mhc_det_train.cake_mhc_train_head`."""

    return CakeMHCTrainHeadFunction.apply(X)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeMHCTrainHeadFunction",
    "cake_mhc_train_head",
    "cake_mhc_train_head_backward",
    "cake_mhc_train_head_forward",
    "is_cake_mhc_train_head_supported",
]
