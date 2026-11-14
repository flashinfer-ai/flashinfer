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

Cake fused SwiGLU + MXFP8 quantization training forward / backward
(SM100 / SM103 / SM107), deterministic.

``gated_input`` ``[M, 2K]`` BF16 holds the gate columns ``[0, K)`` and the up
columns ``[K, 2K)``.  The forward quantizes ``silu(gate) * up`` (``[M, K]``);
the backward quantizes the gate and up gradients (``[M, 2K]``) from
``grad_output`` ``[M, K]``.  Each value is rounded to BF16 and quantized to
E4M3 per 32-element block with an RCEIL E8M0 scale, rowwise and/or colwise,
with the scales in the swizzled 128 x 4 layout.

Every output repeats bit for bit for identical inputs: the op is elementwise
plus a maximum over each 32-element block, which is exact in any order, and
nothing is summed across threads, CTAs or calls.  Every program of a route
writes the same bytes, so the result does not depend on the shape-based
program choice either.

Naming rules:

1. The mode is part of every name.  Every public name of this module carries
   ``det``; the module constants ``ARCHES`` and
   ``SUPPORTED_COMPUTE_CAPABILITIES`` are exempt.  The backend package is
   ``flashinfer.experimental.cake.cake_gated_act_mxfp8_det_train``, imported
   on first use (``import flashinfer`` does not load it).
2. No entry point takes a ``deterministic`` keyword.  No call alerts under
   ``torch.use_deterministic_algorithms(True)``.
3. The former names are gone, with no alias or shim: ``flashinfer.gated_act_mxfp8``
   fails at import, and ``silu_and_mul_mxfp8_quantize`` /
   ``silu_and_mul_mxfp8_quantize_backward`` are attributes of neither
   ``flashinfer`` nor this module.  The fused SwiGLU + MXFP8 section of
   ``docs/api/quantization.rst`` maps each former name to its new name.

Usage::

    row, col, row_scales, col_scales = cake_gated_act_mxfp8_det_train_forward(
        gated_input, rowwise=True, colwise=True
    )
    drow, dcol, drow_scales, dcol_scales = cake_gated_act_mxfp8_det_train_backward(
        gated_input, grad_output, rowwise=True, colwise=True
    )
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any, Tuple

import torch

from .api_logging import flashinfer_api
from .trace.templates.cake_gated_act_mxfp8_det_train import (
    cake_gated_act_mxfp8_det_train_backward_trace_dispatch,
    cake_gated_act_mxfp8_det_train_forward_trace_dispatch,
)
from .utils import register_custom_op, register_fake_op

CakeGatedActMXFP8DetOutputs = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@functools.cache
def _backend() -> Any:
    from .experimental.cake import cake_gated_act_mxfp8_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_gated_act_mxfp8_det_train


# The backend package's architecture constants, resolved on first access by
# the module ``__getattr__`` below.
_BACKEND_OBJECTS = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
)


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        return getattr(_backend(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@functools.cache
def cake_gated_act_mxfp8_det_train_register_custom_ops():
    """The two operations as registered custom ops (``torch.ops.flashinfer``),
    each running the Cake backend."""

    @register_custom_op(
        "flashinfer::cake_gated_act_mxfp8_det_train_forward",
        mutates_args=(),
    )
    def _forward(
        gated_input: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> CakeGatedActMXFP8DetOutputs:
        return _backend().cake_gated_act_mxfp8_det_train_forward(
            gated_input, rowwise=rowwise, colwise=colwise
        )

    @register_fake_op("flashinfer::cake_gated_act_mxfp8_det_train_forward")
    def _fake_forward(
        gated_input: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> CakeGatedActMXFP8DetOutputs:
        backend = _backend()
        return backend.cake_gated_act_mxfp8_det_train_logical_outputs(
            backend.cake_gated_act_mxfp8_det_train_empty_outputs(
                gated_input, backward=False, rowwise=rowwise, colwise=colwise
            )
        )

    @register_custom_op(
        "flashinfer::cake_gated_act_mxfp8_det_train_backward",
        mutates_args=(),
    )
    def _backward(
        gated_input: torch.Tensor,
        grad_output: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> CakeGatedActMXFP8DetOutputs:
        return _backend().cake_gated_act_mxfp8_det_train_backward(
            gated_input, grad_output, rowwise=rowwise, colwise=colwise
        )

    @register_fake_op("flashinfer::cake_gated_act_mxfp8_det_train_backward")
    def _fake_backward(
        gated_input: torch.Tensor,
        grad_output: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> CakeGatedActMXFP8DetOutputs:
        del grad_output
        backend = _backend()
        return backend.cake_gated_act_mxfp8_det_train_logical_outputs(
            backend.cake_gated_act_mxfp8_det_train_empty_outputs(
                gated_input, backward=True, rowwise=rowwise, colwise=colwise
            )
        )

    return SimpleNamespace(forward=_forward, backward=_backward)


@flashinfer_api
def is_cake_gated_act_mxfp8_det_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of an exported architecture.

    Parameters
    ----------
    device : torch.device
        The device to check.

    Returns
    -------
    bool
        ``True`` when the package has kernels for ``device`` (SM100, SM103 or
        SM107).
    """

    return _backend().is_cake_gated_act_mxfp8_det_train_supported(device)


@flashinfer_api(trace=cake_gated_act_mxfp8_det_train_forward_trace_dispatch)
def cake_gated_act_mxfp8_det_train_forward(
    gated_input: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> CakeGatedActMXFP8DetOutputs:
    r"""Apply SwiGLU and emit RCEIL MXFP8 rowwise and/or colwise outputs.

    ``gated_input`` has shape ``[M, 2K]`` and stores gate values followed by
    up values.  The returned tuple is ordered as ``(row_output, col_output,
    row_scales, col_scales)``.  A disabled orientation is represented by
    zero-sized tensors on the input device.  Every output repeats bit for bit
    for identical inputs.

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
    return _backend().cake_gated_act_mxfp8_det_train_forward(
        gated_input, rowwise=rowwise, colwise=colwise
    )


@flashinfer_api(trace=cake_gated_act_mxfp8_det_train_backward_trace_dispatch)
def cake_gated_act_mxfp8_det_train_backward(
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> CakeGatedActMXFP8DetOutputs:
    r"""Apply the SwiGLU backward transform and emit RCEIL MXFP8 outputs.

    The logical result concatenates the gate-input and up-input gradients,
    producing shape ``[M, 2K]`` before quantization.  No intermediate BF16
    result is materialized in global memory.  Every output repeats bit for bit
    for identical inputs.

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
    return _backend().cake_gated_act_mxfp8_det_train_backward(
        gated_input, grad_output, rowwise=rowwise, colwise=colwise
    )


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822
    "CakeGatedActMXFP8DetOutputs",
    "cake_gated_act_mxfp8_det_train_backward",
    "cake_gated_act_mxfp8_det_train_forward",
    "cake_gated_act_mxfp8_det_train_register_custom_ops",
    "is_cake_gated_act_mxfp8_det_train_supported",
]
