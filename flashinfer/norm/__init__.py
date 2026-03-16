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


def _normalize_scale_tensor(
    scale: torch.Tensor, ref_tensor: torch.Tensor
) -> torch.Tensor:
    """Normalize quantization scale tensor to 1D shape (1,) on target device."""
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"scale must be torch.Tensor, got {type(scale)}")
    if scale.device != ref_tensor.device:
        scale = scale.to(ref_tensor.device)
    if scale.dtype != torch.float32:
        scale = scale.to(torch.float32)
    if scale.ndim == 0:
        scale = scale.view(1)
    elif scale.ndim == 1 and scale.numel() == 1:
        pass
    else:
        raise ValueError(
            f"scale must be a scalar tensor or shape (1,), got shape {tuple(scale.shape)}"
        )
    return scale.contiguous()


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
    scale: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Root mean square normalization + fp8 quantization.

    ``out[i] = ((input[i] / RMS(input)) * weight[i]).to(fp8)``

    Parameters
    ----------
    out: torch.Tensor
        The output tensor, will quantize the output to the dtype of this tensor.
    input: torch.Tensor
        Input tensor, 2D shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    scale: torch.Tensor
        Scale factor for quantization, shape (1,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    """
    scale = _normalize_scale_tensor(scale, input)
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
    scale: torch.Tensor,
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
    scale: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Fused add root mean square normalization + fp8 quantization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = ((residual[i] / RMS(residual)) * weight[i]).to(fp8)``

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
    scale: torch.Tensor
        Scale factor for quantization, shape (1,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    scale = _normalize_scale_tensor(scale, input)
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
    scale: torch.Tensor,
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


# cuDNN OSS fused RMSNorm + SiLU engine (SM80+, optimized for VAE shapes on B200)
try:
    from ..cudnn.norm import cudnn_fused_rmsnorm_silu as _cudnn_fused_rmsnorm_silu
except ImportError:
    _cudnn_fused_rmsnorm_silu = None  # type: ignore[misc,assignment]

# Problem sizes with sweep-tuned knob configurations (SM100 / B200).
# Other (C, token) combinations use a conservative fallback heuristic.
_TUNED_C_VALUES = frozenset({64, 128, 160, 256, 320, 512, 640, 1024})
_TUNED_TOKEN_VALUES = frozenset({1560, 6240, 24960, 99840, 399360})


@flashinfer_api
def fused_rmsnorm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Fused RMSNorm + SiLU activation.

    ``out = SiLU(RMSNorm(input, weight, eps))``

    where ``RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight``
    and ``SiLU(y) = y * sigmoid(y)``.

    Uses the cuDNN OSS RmsNormSilu engine on SM80+ GPUs via NVRTC JIT
    compilation.  Requires ``nvidia-cudnn-frontend`` to be installed.

    .. note::

       This kernel is **tuned and optimized for WAN VAE decoder problem
       sizes on B200 (SM100)**:

       - **Optimized C:** {64, 128, 160, 256, 320, 512, 640, 1024}
       - **Optimized token counts:** {1560, 6240, 24960, 99840, 399360}

       On other architectures (A100, H100, L40, etc.) and non-VAE problem
       sizes, a conservative fallback heuristic selects valid kernel
       parameters -- functional but not performance-optimal.

    The output dtype is determined by the ``out`` tensor:
      - ``out.dtype == bfloat16``: standard bf16 output (default)
      - ``out.dtype == float8_e4m3fn``: FP8 quantized output (SM89+)
      - ``out.dtype == uint8`` or ``float4_e2m1fn_x2``: NVFP4 E2M1 packed output (SM100+)

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (num_tokens, hidden_size). Must be bf16.
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,). Must be bf16.
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor. If specified, the kernel will update this tensor
        inplace. The dtype of ``out`` determines the output quantization.

    Returns
    -------
    output: torch.Tensor
        Output tensor, same shape as input. Dtype matches ``out``.

    Raises
    ------
    RuntimeError
        If cuDNN frontend is not available, the GPU is older than SM80,
        or the input dtype/shape is unsupported.
    """
    import warnings
    from ..utils import get_compute_capability

    if not input.is_cuda:
        raise RuntimeError("fused_rmsnorm_silu requires CUDA tensors")

    major, minor = get_compute_capability(input.device)
    sm = major * 10 + minor
    if major < 8:
        raise RuntimeError(
            f"fused_rmsnorm_silu requires SM80+ (Ampere or newer), "
            f"got SM{sm} ({torch.cuda.get_device_name(input.device)})"
        )

    if input.dtype != torch.bfloat16:
        raise RuntimeError(
            f"fused_rmsnorm_silu requires bfloat16 input, got {input.dtype}"
        )

    if input.dim() != 2:
        raise RuntimeError(
            f"fused_rmsnorm_silu requires 2D input (num_tokens, hidden_size), "
            f"got {input.dim()}D"
        )

    if out is None:
        out = torch.empty_like(input)

    _NVFP4_DTYPES = {torch.uint8}
    if hasattr(torch, "float4_e2m1fn_x2"):
        _NVFP4_DTYPES.add(torch.float4_e2m1fn_x2)

    _SUPPORTED_OUT_DTYPES = {torch.bfloat16, torch.float8_e4m3fn} | _NVFP4_DTYPES
    if out.dtype not in _SUPPORTED_OUT_DTYPES:
        raise RuntimeError(
            f"fused_rmsnorm_silu output dtype must be bfloat16, float8_e4m3fn, "
            f"or uint8/float4_e2m1fn_x2 (NVFP4), got {out.dtype}"
        )

    if out.dtype == torch.float8_e4m3fn and sm < 89:
        raise RuntimeError(f"FP8 E4M3 output requires SM89+ (Ada/Hopper), got SM{sm}")

    if out.dtype in _NVFP4_DTYPES and sm < 100:
        raise RuntimeError(f"NVFP4 E2M1 output requires SM100+ (Blackwell), got SM{sm}")

    # Warn once if the problem size or GPU is outside the tuned regime.
    num_tokens, C = input.shape
    is_sm100 = major == 10 and minor == 0
    is_tuned = C in _TUNED_C_VALUES and num_tokens in _TUNED_TOKEN_VALUES
    if not (is_sm100 and is_tuned):
        warnings.warn(
            f"fused_rmsnorm_silu: this kernel is tuned and optimized for "
            f"WAN VAE problem sizes (C in {sorted(_TUNED_C_VALUES)}, "
            f"tokens in {sorted(_TUNED_TOKEN_VALUES)}) on B200 (SM100). "
            f"Current: C={C}, tokens={num_tokens}, "
            f"SM{sm}. Performance may not be optimal.",
            stacklevel=2,
        )

    _cudnn_fused_rmsnorm_silu(input, weight, eps, out=out)
    return out


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
    "fused_rmsnorm_silu",
]
