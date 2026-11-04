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

Backend of :mod:`flashinfer.gated_act_mxfp8`: the Cake fused SwiGLU + MXFP8
quantization training forward / backward (SM100 / SM103 / SM107) as one
self-contained package (host API bodies, the ``cake_jit`` registry, the pinned
``cake_build`` loader and the generated CUDA sources under ``csrc/``).  The
``@flashinfer_api``-decorated entry points and the user-facing documentation
live in :mod:`flashinfer.gated_act_mxfp8`; import that module.

Host design (the knobs are keywords; no environment variable):

- ``gated_input`` ``[M, 2K]`` BF16 holds the gate columns ``[0, K)`` and the
  up columns ``[K, 2K)``.  The forward quantizes ``silu(gate) * up``
  (``[M, K]``), the backward the gate and up gradients (``[M, 2K]``: columns
  ``[0, K)`` the gate gradient, ``[K, 2K)`` the up gradient) from
  ``grad_output`` ``[M, K]``.  Each value is rounded to BF16 and quantized to
  E4M3 per 32-element block with an RCEIL E8M0 scale.
- ``rowwise`` / ``colwise`` select the outputs: ``row_output`` is row-major,
  ``col_output`` column-major (its storage is the transpose), and each scale
  tensor is in the swizzled 128 x 4 layout.  A disabled orientation returns
  zero-size tensors on the input device.
- A call runs one program of its route (``cake_jit.route_name``), chosen by
  ``M * K`` alone (``cake_jit.selected_module``); every program of a route
  writes the same bytes.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    BLOCK,
    INT32_MAX,
    SHAPE_MULTIPLE,
    arch_for_capability,
    route_name,
    run_module,
    selected_module,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
# The kernels load ``gated_input`` and ``grad_output`` in 32-byte rows.
ALIGNMENT_BYTES = 32

GatedActMxfp8Outputs = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _arch_of_index(device_index: int) -> Optional[str]:
    """The exported architecture of one CUDA device index, queried once per process."""

    return arch_for_capability(tuple(torch.cuda.get_device_capability(device_index)))


def is_cake_gated_act_mxfp8_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of an exported architecture."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    return _arch_of_index(_device_index(device)) is not None


def check_inputs(
    gated_input: torch.Tensor,
    grad_output: Optional[torch.Tensor],
    *,
    rowwise: bool,
    colwise: bool,
) -> tuple[int, int]:
    """Validate one call and return ``(M, K)``.

    ``gated_input`` is a contiguous, 32-byte aligned BF16 CUDA tensor ``[M, 2K]``
    with ``M`` and ``K`` positive multiples of 128 and ``M * 2K - 1`` (the last
    element of the input and of the backward outputs) within signed 32 bits;
    ``grad_output`` (the backward) is a contiguous, aligned BF16 tensor
    ``[M, K]`` on the same device; at least one of ``rowwise`` and ``colwise``
    is set; and the device is SM100, SM103 or SM107.
    """

    if not isinstance(gated_input, torch.Tensor) or not gated_input.is_cuda:
        raise ValueError("gated_input must be a CUDA tensor")
    if gated_input.dtype is not torch.bfloat16:
        raise TypeError("gated_input must have bfloat16 dtype")
    if gated_input.ndim != 2 or not gated_input.is_contiguous():
        raise ValueError("gated_input must be contiguous with shape [M, 2K]")
    m, doubled_k = map(int, gated_input.shape)
    if doubled_k % 2:
        raise ValueError("gated_input.shape[1] must be even")
    k = doubled_k // 2
    if m <= 0 or k <= 0 or m % SHAPE_MULTIPLE or k % SHAPE_MULTIPLE:
        raise ValueError("M and K must be positive multiples of 128")
    if m * doubled_k - 1 > INT32_MAX:
        raise ValueError("gated activation shape exceeds signed int32 indexing")
    if gated_input.data_ptr() % ALIGNMENT_BYTES:
        raise ValueError("gated_input must be 32-byte aligned")
    if not rowwise and not colwise:
        raise ValueError("at least one of rowwise or colwise must be enabled")
    if _arch_of_index(_device_index(gated_input.device)) is None:
        raise RuntimeError(
            "fused gated MXFP8 quantization requires SM100, SM103 or SM107"
        )
    if grad_output is not None:
        if not isinstance(grad_output, torch.Tensor) or not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        if grad_output.dtype is not torch.bfloat16:
            raise TypeError("grad_output must have bfloat16 dtype")
        if tuple(grad_output.shape) != (m, k) or not grad_output.is_contiguous():
            raise ValueError("grad_output must be contiguous with shape [M, K]")
        if grad_output.device != gated_input.device:
            raise ValueError("gated_input and grad_output must be on the same device")
        if grad_output.data_ptr() % ALIGNMENT_BYTES:
            raise ValueError("grad_output must be 32-byte aligned")
    return m, k


def empty_outputs(
    gated_input: torch.Tensor,
    *,
    backward: bool,
    rowwise: bool,
    colwise: bool,
) -> GatedActMxfp8Outputs:
    """The four output tensors of one call, unwritten: ``(row_output, col_output,
    row_scales, col_scales)``, ``[M, N]`` E4M3 row-major and column-major values
    and their swizzled scale bytes (``N`` = ``K`` forward, ``2K`` backward), a
    disabled orientation's pair zero-size on the input device."""

    m = int(gated_input.shape[0])
    n = int(gated_input.shape[1]) if backward else int(gated_input.shape[1]) // 2
    device = gated_input.device
    fp8 = torch.float8_e4m3fn
    row_output = (
        torch.empty_strided((m, n), (n, 1), dtype=fp8, device=device)
        if rowwise
        else gated_input.new_empty(0, dtype=fp8)
    )
    col_output = (
        torch.empty_strided((m, n), (1, m), dtype=fp8, device=device)
        if colwise
        else gated_input.new_empty(0, dtype=fp8)
    )
    row_scales = (
        torch.empty((m, n // BLOCK), dtype=torch.uint8, device=device)
        if rowwise
        else gated_input.new_empty(0, dtype=torch.uint8)
    )
    col_scales = (
        torch.empty(n * (m // BLOCK), dtype=torch.uint8, device=device)
        if colwise
        else gated_input.new_empty(0, dtype=torch.uint8)
    )
    return row_output, col_output, row_scales, col_scales


def logical_outputs(outputs: GatedActMxfp8Outputs) -> GatedActMxfp8Outputs:
    """``outputs`` as the public API returns them: the scale bytes as E8M0."""

    row_output, col_output, row_scales, col_scales = outputs
    return (
        row_output,
        col_output,
        row_scales.view(torch.float8_e8m0fnu),
        col_scales.view(torch.float8_e8m0fnu),
    )


def _run(
    gated_input: torch.Tensor,
    grad_output: Optional[torch.Tensor],
    outputs: GatedActMxfp8Outputs,
    *,
    M: int,
    K: int,
    rowwise: bool,
    colwise: bool,
) -> None:
    device_index = _device_index(gated_input.device)
    arch = _arch_of_index(device_index)
    kind = route_name(
        backward=grad_output is not None, rowwise=rowwise, colwise=colwise
    )
    name = selected_module(arch, kind, M=M, K=K)
    row_output, col_output, row_scales, col_scales = outputs
    values = {
        "gated_input": gated_input,
        "row_output": row_output.view(torch.uint8),
        # The column-major values are stored as their [N, M] transpose.
        "col_output": (col_output.t() if colwise else col_output).view(torch.uint8),
        "row_scales": row_scales,
        "col_scales": col_scales,
        "M": M,
        "K": K,
    }
    if grad_output is not None:
        values["grad_output"] = grad_output
    run_module(name, values, device_index=device_index)


def silu_and_mul_mxfp8_quantize(
    gated_input: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> GatedActMxfp8Outputs:
    """SwiGLU of ``gated_input`` ``[M, 2K]`` quantized to MXFP8 (RCEIL scales):
    ``(row_output, col_output, row_scales, col_scales)``, a disabled
    orientation's pair zero-size."""

    m, k = check_inputs(gated_input, None, rowwise=rowwise, colwise=colwise)
    outputs = empty_outputs(
        gated_input, backward=False, rowwise=rowwise, colwise=colwise
    )
    _run(gated_input, None, outputs, M=m, K=k, rowwise=rowwise, colwise=colwise)
    return logical_outputs(outputs)


def silu_and_mul_mxfp8_quantize_backward(
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> GatedActMxfp8Outputs:
    """The SwiGLU backward of ``gated_input`` ``[M, 2K]`` and ``grad_output``
    ``[M, K]``, its gate and up gradients ``[M, 2K]`` quantized to MXFP8:
    ``(row_output, col_output, row_scales, col_scales)``, a disabled
    orientation's pair zero-size."""

    if grad_output is None:
        raise ValueError("grad_output must be a CUDA tensor")
    m, k = check_inputs(gated_input, grad_output, rowwise=rowwise, colwise=colwise)
    outputs = empty_outputs(
        gated_input, backward=True, rowwise=rowwise, colwise=colwise
    )
    _run(gated_input, grad_output, outputs, M=m, K=k, rowwise=rowwise, colwise=colwise)
    return logical_outputs(outputs)


__all__ = [
    "ALIGNMENT_BYTES",
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "GatedActMxfp8Outputs",
    "check_inputs",
    "empty_outputs",
    "is_cake_gated_act_mxfp8_supported",
    "logical_outputs",
    "silu_and_mul_mxfp8_quantize",
    "silu_and_mul_mxfp8_quantize_backward",
]
