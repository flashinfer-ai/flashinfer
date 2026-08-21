"""Fused gated activation and MXFP8 quantization for Blackwell GPUs."""

import functools
from types import SimpleNamespace
from typing import Tuple

import torch

from .api_logging import flashinfer_api
from .jit.gated_act_mxfp8 import gen_gated_act_mxfp8_module
from .trace.templates.gated_act_mxfp8 import (
    silu_and_mul_mxfp8_backward_trace_dispatch,
    silu_and_mul_mxfp8_forward_trace_dispatch,
)
from .utils import get_compute_capability, register_custom_op, register_fake_op


GatedActMxfp8Outputs = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _validate_inputs(
    gated_input: torch.Tensor,
    grad_output: torch.Tensor | None,
    *,
    rowwise: bool,
    colwise: bool,
) -> tuple[int, int]:
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
    if m <= 0 or k <= 0 or m % 128 or k % 128:
        raise ValueError("M and K must be positive multiples of 128")
    if m * doubled_k - k - 1 > 2**31 - 1:
        raise ValueError("gated activation shape exceeds signed int32 indexing")
    if gated_input.data_ptr() % 32:
        raise ValueError("gated_input must be 32-byte aligned")
    if not rowwise and not colwise:
        raise ValueError("at least one of rowwise or colwise must be enabled")
    major, minor = get_compute_capability(gated_input.device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise RuntimeError("fused gated MXFP8 quantization requires SM100 or SM103")
    if grad_output is not None:
        if not isinstance(grad_output, torch.Tensor) or not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        if grad_output.dtype is not torch.bfloat16:
            raise TypeError("grad_output must have bfloat16 dtype")
        if tuple(grad_output.shape) != (m, k) or not grad_output.is_contiguous():
            raise ValueError("grad_output must be contiguous with shape [M, K]")
        if grad_output.device != gated_input.device:
            raise ValueError("gated_input and grad_output must be on the same device")
        if grad_output.data_ptr() % 32:
            raise ValueError("grad_output must be 32-byte aligned")
    return m, k


def _allocate_outputs(
    gated_input: torch.Tensor,
    output_k: int,
    rowwise: bool,
    colwise: bool,
) -> GatedActMxfp8Outputs:
    m = int(gated_input.shape[0])
    row_output = (
        torch.empty_strided(
            (m, output_k),
            (output_k, 1),
            dtype=torch.float8_e4m3fn,
            device=gated_input.device,
        )
        if rowwise
        else gated_input.new_empty(0, dtype=torch.float8_e4m3fn)
    )
    col_output = (
        torch.empty_strided(
            (m, output_k),
            (1, m),
            dtype=torch.float8_e4m3fn,
            device=gated_input.device,
        )
        if colwise
        else gated_input.new_empty(0, dtype=torch.float8_e4m3fn)
    )
    row_scales = (
        torch.empty(
            (m, output_k // 32),
            dtype=torch.uint8,
            device=gated_input.device,
        )
        if rowwise
        else gated_input.new_empty(0, dtype=torch.uint8)
    )
    col_scales = (
        torch.empty(
            output_k * (m // 32),
            dtype=torch.uint8,
            device=gated_input.device,
        )
        if colwise
        else gated_input.new_empty(0, dtype=torch.uint8)
    )
    return row_output, col_output, row_scales, col_scales


def _logical_outputs(outputs: GatedActMxfp8Outputs) -> GatedActMxfp8Outputs:
    row_output, col_output, row_scales, col_scales = outputs
    return (
        row_output,
        col_output,
        row_scales.view(torch.float8_e8m0fnu),
        col_scales.view(torch.float8_e8m0fnu),
    )


@functools.cache
def get_gated_act_mxfp8_module():
    module = gen_gated_act_mxfp8_module().build_and_load()

    @register_custom_op(
        "flashinfer::silu_and_mul_mxfp8_quantize",
        mutates_args=(),
    )
    def _forward(
        gated_input: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> GatedActMxfp8Outputs:
        outputs = _allocate_outputs(
            gated_input,
            int(gated_input.shape[1]) // 2,
            rowwise,
            colwise,
        )
        module.forward(gated_input, *outputs, rowwise, colwise)
        return _logical_outputs(outputs)

    @register_fake_op("flashinfer::silu_and_mul_mxfp8_quantize")
    def _fake_forward(
        gated_input: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> GatedActMxfp8Outputs:
        return _logical_outputs(
            _allocate_outputs(
                gated_input,
                int(gated_input.shape[1]) // 2,
                rowwise,
                colwise,
            )
        )

    @register_custom_op(
        "flashinfer::silu_and_mul_mxfp8_quantize_backward",
        mutates_args=(),
    )
    def _backward(
        gated_input: torch.Tensor,
        grad_output: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> GatedActMxfp8Outputs:
        outputs = _allocate_outputs(
            gated_input,
            int(gated_input.shape[1]),
            rowwise,
            colwise,
        )
        module.backward(gated_input, grad_output, *outputs, rowwise, colwise)
        return _logical_outputs(outputs)

    @register_fake_op("flashinfer::silu_and_mul_mxfp8_quantize_backward")
    def _fake_backward(
        gated_input: torch.Tensor,
        grad_output: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> GatedActMxfp8Outputs:
        del grad_output
        return _logical_outputs(
            _allocate_outputs(
                gated_input,
                int(gated_input.shape[1]),
                rowwise,
                colwise,
            )
        )

    return SimpleNamespace(forward=_forward, backward=_backward)


@flashinfer_api(trace=silu_and_mul_mxfp8_forward_trace_dispatch)
def silu_and_mul_mxfp8_quantize(
    gated_input: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> GatedActMxfp8Outputs:
    r"""Apply SwiGLU and emit RCEIL MXFP8 rowwise and/or colwise outputs.

    ``gated_input`` has shape ``[M, 2K]`` and stores gate values followed by
    up values.  The returned tuple is ordered as ``(row_output, col_output,
    row_scales, col_scales)``.  A disabled orientation is represented by
    zero-sized tensors on the input device.
    """
    _validate_inputs(gated_input, None, rowwise=rowwise, colwise=colwise)
    return get_gated_act_mxfp8_module().forward(gated_input, rowwise, colwise)


@flashinfer_api(trace=silu_and_mul_mxfp8_backward_trace_dispatch)
def silu_and_mul_mxfp8_quantize_backward(
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> GatedActMxfp8Outputs:
    r"""Apply the SwiGLU backward transform and emit RCEIL MXFP8 outputs.

    The logical result concatenates the gate-input and up-input gradients,
    producing shape ``[M, 2K]`` before quantization.  No intermediate BF16
    result is materialized in global memory.
    """
    _validate_inputs(
        gated_input,
        grad_output,
        rowwise=rowwise,
        colwise=colwise,
    )
    return get_gated_act_mxfp8_module().backward(
        gated_input,
        grad_output,
        rowwise,
        colwise,
    )


__all__ = [
    "GatedActMxfp8Outputs",
    "get_gated_act_mxfp8_module",
    "silu_and_mul_mxfp8_quantize",
    "silu_and_mul_mxfp8_quantize_backward",
]
