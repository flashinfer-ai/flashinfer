"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional

import torch

from .jit import FLASHINFER_CSRC_DIR, has_prebuilt_ops, load_cuda_ops
from .utils import get_cuda_stream, register_custom_op, register_fake_op

_norm_module = None


def get_norm_module():
    global _norm_module
    if _norm_module is None:
        if has_prebuilt_ops:
            from . import _kernels

            _norm_module = _kernels
        else:
            _norm_module = load_cuda_ops(
                "norm",
                [
                    FLASHINFER_CSRC_DIR / "norm.cu",
                    FLASHINFER_CSRC_DIR / "flashinfer_norm_ops.cu",
                ],
            )
    return _norm_module


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm(out, input, weight, eps)
    return out


@register_custom_op("flashinfer::rmsnorm", mutates_args=("out",))
def _rmsnorm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    with input.device as device:  # device guard
        get_norm_module().rmsnorm(out, input, weight, eps, get_cuda_stream(device))


@register_fake_op("flashinfer::rmsnorm")
def _rmsnorm_fake(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    pass


@register_custom_op("flashinfer::fused_add_rmsnorm", mutates_args=("input", "residual"))
def fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    r"""Fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    """
    with input.device as device:  # device guard
        get_norm_module().fused_add_rmsnorm(
            input, residual, weight, eps, get_cuda_stream(device)
        )


@register_fake_op("flashinfer::fused_add_rmsnorm")
def _fused_add_rmsnorm_fake(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    pass


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Gemma-style root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * (weight[i] + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    _gemma_rmsnorm(out, input, weight, eps)
    return out


@register_custom_op("flashinfer::gemma_rmsnorm", mutates_args=("out",))
def _gemma_rmsnorm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    with input.device as device:  # device guard
        get_norm_module().gemma_rmsnorm(
            out, input, weight, eps, get_cuda_stream(device)
        )


@register_fake_op("flashinfer::gemma_rmsnorm")
def _gemma_rmsnorm_fake(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    pass


@register_custom_op(
    "flashinfer::gemma_fused_add_rmsnorm", mutates_args=("input", "residual")
)
def gemma_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    r"""Gemma-style fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * (weight + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    """
    with input.device as device:
        get_norm_module().gemma_fused_add_rmsnorm(
            input, residual, weight, eps, get_cuda_stream(device)
        )


@register_fake_op("flashinfer::gemma_fused_add_rmsnorm")
def _gemma_fused_add_rmsnorm_fake(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    pass
