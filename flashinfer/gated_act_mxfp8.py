"""Fused gated activation and MXFP8 quantization for Blackwell GPUs."""

import functools
from types import SimpleNamespace
from typing import Tuple

import torch

from . import cake_gated_act_mxfp8_backend as _backend
from .api_logging import flashinfer_api
from .trace.templates.gated_act_mxfp8 import (
    silu_and_mul_mxfp8_backward_trace_dispatch,
    silu_and_mul_mxfp8_forward_trace_dispatch,
)
from .utils import register_custom_op, register_fake_op


GatedActMxfp8Outputs = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@functools.cache
def get_gated_act_mxfp8_module():
    """The two operations as registered custom ops (``torch.ops.flashinfer``),
    each running the Cake backend."""

    @register_custom_op(
        "flashinfer::silu_and_mul_mxfp8_quantize",
        mutates_args=(),
    )
    def _forward(
        gated_input: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> GatedActMxfp8Outputs:
        return _backend.silu_and_mul_mxfp8_quantize(
            gated_input, rowwise=rowwise, colwise=colwise
        )

    @register_fake_op("flashinfer::silu_and_mul_mxfp8_quantize")
    def _fake_forward(
        gated_input: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> GatedActMxfp8Outputs:
        return _backend.logical_outputs(
            _backend.empty_outputs(
                gated_input, backward=False, rowwise=rowwise, colwise=colwise
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
        return _backend.silu_and_mul_mxfp8_quantize_backward(
            gated_input, grad_output, rowwise=rowwise, colwise=colwise
        )

    @register_fake_op("flashinfer::silu_and_mul_mxfp8_quantize_backward")
    def _fake_backward(
        gated_input: torch.Tensor,
        grad_output: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> GatedActMxfp8Outputs:
        del grad_output
        return _backend.logical_outputs(
            _backend.empty_outputs(
                gated_input, backward=True, rowwise=rowwise, colwise=colwise
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

    Parameters
    ----------
    gated_input : torch.Tensor
        Contiguous BF16 CUDA tensor with shape ``[M, 2K]``.
    rowwise : bool, optional
        Whether to emit rowwise MXFP8 values and scales.
    colwise : bool, optional
        Whether to emit colwise MXFP8 values and scales.

    Returns
    -------
    tuple of torch.Tensor
        Rowwise values, colwise values, rowwise scales, and colwise scales.
        Outputs for disabled orientations are zero-sized tensors.
    """
    return _backend.silu_and_mul_mxfp8_quantize(
        gated_input, rowwise=rowwise, colwise=colwise
    )


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

    Parameters
    ----------
    gated_input : torch.Tensor
        Contiguous BF16 CUDA tensor with shape ``[M, 2K]``.
    grad_output : torch.Tensor
        Contiguous BF16 CUDA tensor with shape ``[M, K]``.
    rowwise : bool, optional
        Whether to emit rowwise MXFP8 values and scales.
    colwise : bool, optional
        Whether to emit colwise MXFP8 values and scales.

    Returns
    -------
    tuple of torch.Tensor
        Rowwise values, colwise values, rowwise scales, and colwise scales for
        the concatenated gate-input and up-input gradients. Outputs for
        disabled orientations are zero-sized tensors.
    """
    return _backend.silu_and_mul_mxfp8_quantize_backward(
        gated_input, grad_output, rowwise=rowwise, colwise=colwise
    )


__all__ = [
    "GatedActMxfp8Outputs",
    "get_gated_act_mxfp8_module",
    "silu_and_mul_mxfp8_quantize",
    "silu_and_mul_mxfp8_quantize_backward",
]
