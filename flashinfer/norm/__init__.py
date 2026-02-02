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

FlashInfer Normalization Kernels
================================

This package provides high-performance normalization kernels:

- RMSNorm: Root Mean Square Normalization
- LayerNorm: Layer Normalization
- Fused Add + RMSNorm: Combined residual add and RMSNorm
- Quantized variants with FP8/FP4 output
"""

import functools
import os
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..utils import device_support_pdl, register_custom_op, register_fake_op

# Always import gen_norm_module for JIT warmup and CUDA fallback
from ..jit.norm import gen_norm_module

# Use CUDA JIT implementation instead of CuTe DSL (for debugging/fallback)
# Also fallback to CUDA JIT if nvidia-cutlass-dsl is not installed
_USE_CUDA_NORM = os.environ.get("FLASHINFER_USE_CUDA_NORM", "0") == "1"

if not _USE_CUDA_NORM:
    try:
        from .kernels import (
            rmsnorm_cute,
            qk_rmsnorm_cute,
            rmsnorm_quant_cute,
            fused_add_rmsnorm_cute,
            fused_add_rmsnorm_quant_cute,
            layernorm_cute,
        )
    except (ImportError, AttributeError):
        # nvidia-cutlass-dsl not installed or incompatible version
        _USE_CUDA_NORM = True

if _USE_CUDA_NORM:

    @functools.cache
    def get_norm_module():
        return gen_norm_module().build_and_load()


@flashinfer_api
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm_impl(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::rmsnorm", mutates_args=("out",))
def _rmsnorm_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if _USE_CUDA_NORM:
        get_norm_module().rmsnorm(out, input, weight, eps, enable_pdl)
    else:
        if input.dim() == 3:
            qk_rmsnorm_cute(
                input, weight, out, eps, weight_bias=0.0, enable_pdl=enable_pdl
            )
        else:
            rmsnorm_cute(
                input, weight, out, eps, weight_bias=0.0, enable_pdl=enable_pdl
            )


@register_fake_op("flashinfer::rmsnorm")
def _rmsnorm_impl_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api
@register_custom_op("flashinfer::rmsnorm_quant", mutates_args=("out",))
def rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    out: torch.Tensor
        The output tensor, will quantize the output to the dtype of this tensor.
    input: torch.Tensor
        Input tensor, 2D shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    scale: float
        Scale factor for quantization.
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, 2D shape (batch_size, hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if _USE_CUDA_NORM:
        get_norm_module().rmsnorm_quant(out, input, weight, scale, eps, enable_pdl)
    else:
        rmsnorm_quant_cute(
            out, input, weight, scale, eps, weight_bias=0.0, enable_pdl=enable_pdl
        )


@register_fake_op("flashinfer::rmsnorm_quant")
def _rmsnorm_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api
@register_custom_op("flashinfer::fused_add_rmsnorm", mutates_args=("input", "residual"))
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
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
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if _USE_CUDA_NORM:
        get_norm_module().fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)
    else:
        fused_add_rmsnorm_cute(
            input, residual, weight, eps, weight_bias=0.0, enable_pdl=enable_pdl
        )


@register_fake_op("flashinfer::fused_add_rmsnorm")
def _fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api
@register_custom_op(
    "flashinfer::fused_add_rmsnorm_quant", mutates_args=("out", "residual")
)
def fused_add_rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    out: torch.Tensor
        The output tensor, will quantize the output to the dtype of this tensor.
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    scale: float
        Scale factor for quantization.
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if _USE_CUDA_NORM:
        get_norm_module().fused_add_rmsnorm_quant(
            out, input, residual, weight, scale, eps, enable_pdl
        )
    else:
        fused_add_rmsnorm_quant_cute(
            out,
            input,
            residual,
            weight,
            scale,
            eps,
            weight_bias=0.0,
            enable_pdl=enable_pdl,
        )


@register_fake_op("flashinfer::fused_add_rmsnorm_quant")
def _fused_add_rmsnorm_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api
def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
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
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    _gemma_rmsnorm_impl(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::gemma_rmsnorm", mutates_args=("out",))
def _gemma_rmsnorm_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if _USE_CUDA_NORM:
        get_norm_module().gemma_rmsnorm(out, input, weight, eps, enable_pdl)
    else:
        if input.dim() == 3:
            qk_rmsnorm_cute(
                input, weight, out, eps, weight_bias=1.0, enable_pdl=enable_pdl
            )
        else:
            rmsnorm_cute(
                input, weight, out, eps, weight_bias=1.0, enable_pdl=enable_pdl
            )


@register_fake_op("flashinfer::gemma_rmsnorm")
def _gemma_rmsnorm_impl_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api
@register_custom_op(
    "flashinfer::gemma_fused_add_rmsnorm", mutates_args=("input", "residual")
)
def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
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
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if _USE_CUDA_NORM:
        get_norm_module().gemma_fused_add_rmsnorm(
            input, residual, weight, eps, enable_pdl
        )
    else:
        fused_add_rmsnorm_cute(
            input, residual, weight, eps, weight_bias=1.0, enable_pdl=enable_pdl
        )


@register_fake_op("flashinfer::gemma_fused_add_rmsnorm")
def _gemma_fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api
@register_custom_op("flashinfer::layernorm", mutates_args=())
def layernorm(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Layer normalization.
    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size). Need to be bfloat16.
    gemma: torch.Tensor
        Gemma tensor, shape (hidden_size,). Need to be float32.
    beta: torch.Tensor
        Beta tensor, shape (hidden_size,). Need to be float32.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    output: torch.Tensor
        Layer Normalized tensor, shape (batch_size, hidden_size). Same dtype as input.
    """
    out = torch.empty_like(input)
    if _USE_CUDA_NORM:
        get_norm_module().layernorm(out, input, gemma, beta, eps)
    else:
        layernorm_cute(out, input, gemma, beta, eps)
    return out


@register_fake_op("flashinfer::layernorm")
def _layernorm_fake(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    b, k = input.shape
    return input.new_empty([b, k])


# CuTe-DSL fused RMSNorm + FP4 Quantization kernels
# These require SM100+ (Blackwell) GPUs and nvidia-cutlass-dsl
try:
    from ..cute_dsl import rmsnorm_fp4quant as rmsnorm_fp4quant
    from ..cute_dsl import add_rmsnorm_fp4quant as add_rmsnorm_fp4quant
except ImportError:
    # nvidia-cutlass-dsl not installed, these functions will not be available
    pass


# Public API exports
__all__ = [
    # JIT module generator (always available)
    "gen_norm_module",
    # Public APIs
    "rmsnorm",
    "rmsnorm_quant",
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_quant",
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "layernorm",
]
