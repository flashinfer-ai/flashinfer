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
import warnings
from typing import Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.norm import (
    fused_add_rmsnorm_quant_trace,
    fused_add_rmsnorm_trace,
    fused_rmsnorm_silu_trace,
    gemma_fused_add_rmsnorm_trace,
    gemma_rmsnorm_trace,
    layernorm_trace,
    rmsnorm_quant_trace,
    rmsnorm_trace,
)
from ..utils import (
    backend_requirement,
    device_support_pdl,
    get_compute_capability,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

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


@functools.cache
def get_norm_module():
    """Get or compile the CUDA JIT norm module.

    Always available regardless of _USE_CUDA_NORM setting, since some
    fused kernels (e.g. fused_qk_rmsnorm_rope) only have a CUDA JIT
    implementation and no CuTe DSL alternative.
    """
    return gen_norm_module().build_and_load()


def _normalize_scale_tensor(
    scale: Union[float, torch.Tensor], ref_tensor: torch.Tensor
) -> torch.Tensor:
    """Normalize quantization scale to 1D tensor of shape (1,) on target device."""
    if not isinstance(scale, torch.Tensor):
        warnings.warn(
            "Passing scale as a float is deprecated and will be removed in a future "
            "release. Use a torch.Tensor of shape (1,) instead.",
            FutureWarning,
            stacklevel=3,
        )
        scale = torch.tensor([scale], dtype=torch.float32, device=ref_tensor.device)
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


@flashinfer_api(trace=rmsnorm_trace)
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
    if enable_pdl is None or enable_pdl:
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


@flashinfer_api(trace=rmsnorm_quant_trace)
@register_custom_op("flashinfer::rmsnorm_quant", mutates_args=("out",))
def rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: Union[float, torch.Tensor],
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
    if enable_pdl is None or enable_pdl:
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


@flashinfer_api(trace=fused_add_rmsnorm_trace)
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
    if enable_pdl is None or enable_pdl:
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


@flashinfer_api(trace=fused_add_rmsnorm_quant_trace)
@register_custom_op(
    "flashinfer::fused_add_rmsnorm_quant", mutates_args=("out", "residual")
)
def fused_add_rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: Union[float, torch.Tensor],
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
    if enable_pdl is None or enable_pdl:
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


@flashinfer_api(trace=gemma_rmsnorm_trace)
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
    if enable_pdl is None or enable_pdl:
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


@flashinfer_api(trace=gemma_fused_add_rmsnorm_trace)
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
    if enable_pdl is None or enable_pdl:
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


@flashinfer_api(trace=layernorm_trace)
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


# ============================================================
# Fused RMSNorm + SiLU kernel (SM100 optimized)
# ============================================================

from ..jit.rmsnorm_silu import (
    gen_rmsnorm_silu_module,
    select_knobs,
    _estimate_ctas_per_row,
)


@functools.cache
def _get_rmsnorm_silu_sm_count(device_id: int):
    """Cache the SM count per device."""
    props = torch.cuda.get_device_properties(device_id)
    return props.multi_processor_count


@functools.cache
def _get_rmsnorm_silu_module(
    C, output_dtype, warps_m, ctas_per_row, bytes_per_ldg, kernel_cfg, occupancy
):
    return gen_rmsnorm_silu_module(
        C, output_dtype, warps_m, ctas_per_row, bytes_per_ldg, kernel_cfg, occupancy
    ).build_and_load()


def _compute_rmsnorm_silu_workspace_size(
    rows, cols, output_dtype, warps_m, ctas_per_row, kernel_cfg, occupancy, sm_count
):
    """Compute workspace size matching the engine's layout."""
    # rs
    ws = rows * 4  # sizeof(float)
    ws = ((ws + 127) // 128) * 128
    # fp8_scale
    ws += 4
    ws = ((ws + 127) // 128) * 128
    # cooperative workspace (multi-CTA)
    if ctas_per_row > 1:
        ctas_per_col_max = (rows + warps_m - 1) // warps_m
        if kernel_cfg == 2:
            ctas_per_col = ctas_per_col_max
        else:
            ctas_per_col = min(sm_count * occupancy // ctas_per_row, ctas_per_col_max)
        ctas_per_col = max(ctas_per_col, 1)
        ws += ctas_per_col * warps_m * ctas_per_row * 8 * 2  # sizeof(float2) * 2
        ws = ((ws + 127) // 128) * 128
        ws += 2 * ctas_per_col * 4  # sizeof(int32_t)
        ws = ((ws + 127) // 128) * 128
    ws += 128  # final alignment padding
    return ws


def _torch_dtype_to_str(dtype):
    if dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float8_e4m3fn:
        return "fp8"
    elif hasattr(torch, "float4_e2m1fn_x2") and dtype == torch.float4_e2m1fn_x2:
        return "nvfp4"
    raise ValueError(
        "Unsupported output dtype for fused_rmsnorm_silu: "
        f"{dtype}. Supported dtypes: bfloat16, float8_e4m3fn, float4_e2m1fn_x2"
    )


@flashinfer_api(trace=fused_rmsnorm_silu_trace)
def fused_rmsnorm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    block_scale: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple]:
    r"""Fused RMSNorm + SiLU activation.

    ``out[i] = SiLU(RMSNorm(input[i], weight, eps))``

    where ``SiLU(x) = x / (1 + exp(-x))``

    Optimized for SM100 (B200) for WAN VAE decoder problem sizes.
    Other shapes and architectures (SM80+) use conservative fallback heuristics.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape ``(num_tokens, hidden_size)``, dtype ``bfloat16``.
    weight: torch.Tensor
        Scale (gamma) tensor, shape ``(hidden_size,)``, dtype ``bfloat16``.
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        Output tensor. If ``None``, allocated as ``bfloat16`` matching input.
        The dtype of ``out`` selects the output format:

        - ``torch.bfloat16``: shape ``(num_tokens, hidden_size)``.
        - ``torch.float8_e4m3fn``: FP8 E4M3 output, shape ``(num_tokens, hidden_size)``.
          Requires SM89+ (Ada/Hopper).
        - ``torch.float4_e2m1fn_x2``: NVFP4 block-scaled output, shape
          ``(num_tokens, hidden_size // 2)``. Requires SM100+ (Blackwell)
          and ``hidden_size`` divisible by 16.
    block_scale: Optional[torch.Tensor]
        Pre-allocated output tensor for per-block scale factors (NVFP4 only).
        Shape ``(num_tokens, hidden_size // 16)``, dtype ``torch.float8_e4m3fn``.
        If ``None``, allocated automatically when ``out`` is NVFP4.
        Ignored for bf16/fp8 output.

    Returns
    -------
    output: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        For bf16/fp8: normalized + SiLU activated tensor,
        shape ``(num_tokens, hidden_size)``.

        For NVFP4: a tuple ``(y_fp4, block_scale)`` following the same
        convention as :func:`rmsnorm_fp4quant`. ``y_fp4`` has shape
        ``(num_tokens, hidden_size // 2)`` with dtype ``float4_e2m1fn_x2``,
        and ``block_scale`` has shape ``(num_tokens, hidden_size // 16)``
        with dtype ``float8_e4m3fn`` (one E4M3 scale per 16-element block).

    Notes
    -----
    Kernel tuning knobs are sweep-optimized on B200 (SM100) for WAN VAE
    decoder problem sizes: ``hidden_size`` in {64, 128, 160, 256, 320, 512,
    640, 1024} and ``num_tokens`` in {1560, 6240, 24960, 99840, 399360}.
    Other problem sizes use conservative fallback heuristics that are
    functionally correct but may not achieve peak throughput. Performance
    on non-SM100 architectures uses the same fallback path.
    """
    if input.device.type != "cuda":
        raise ValueError("fused_rmsnorm_silu requires CUDA tensors")
    if input.dtype != torch.bfloat16:
        raise ValueError(f"input must be torch.bfloat16, got {input.dtype}")
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"weight must be torch.bfloat16, got {weight.dtype}")
    if input.ndim != 2:
        raise ValueError(
            f"input must be 2D [num_tokens, hidden_size], got ndim={input.ndim}"
        )
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1D [hidden_size], got ndim={weight.ndim}")
    if weight.device != input.device:
        raise ValueError("weight must be on the same device as input")

    if out is None:
        out = torch.empty_like(input)
    if out.device != input.device:
        raise ValueError("out must be on the same device as input")

    num_tokens = input.size(0)
    C = input.size(1)
    if weight.size(0) != C:
        raise ValueError(
            f"weight shape mismatch: expected [{C}], got {tuple(weight.shape)}"
        )
    output_dtype_str = _torch_dtype_to_str(out.dtype)

    if output_dtype_str in ("bf16", "fp8"):
        if tuple(out.shape) != tuple(input.shape):
            raise ValueError(
                f"out shape mismatch for {output_dtype_str}: expected {tuple(input.shape)}, got {tuple(out.shape)}"
            )
    elif output_dtype_str == "nvfp4":
        expected_shape = (num_tokens, C // 2)
        if C % 16 != 0:
            raise ValueError(
                f"nvfp4 output requires hidden_size divisible by 16, got C={C}"
            )
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out shape mismatch for nvfp4: expected {expected_shape}, got {tuple(out.shape)}"
            )

    major, minor = get_compute_capability(input.device)
    sm_version = major * 10 + minor
    if sm_version < 80:
        raise RuntimeError("fused_rmsnorm_silu requires SM80+")
    if output_dtype_str == "fp8" and sm_version < 89:
        raise RuntimeError("FP8 output requires SM89+ (Ada/Hopper)")
    if output_dtype_str == "nvfp4" and sm_version < 100:
        raise RuntimeError("NVFP4 output requires SM100+ (Blackwell)")

    knobs = select_knobs(C, num_tokens, output_dtype_str, sm_version)
    if knobs is None:
        raise ValueError(
            f"Unsupported problem size for fused_rmsnorm_silu: "
            f"C={C}, num_tokens={num_tokens}, dtype={output_dtype_str}"
        )

    warps_m, split_cols, kernel_cfg, occupancy, bytes_per_ldg = knobs
    ctas_per_row = _estimate_ctas_per_row(C, split_cols, kernel_cfg, bytes_per_ldg)
    sm_count = _get_rmsnorm_silu_sm_count(input.device.index)

    module = _get_rmsnorm_silu_module(
        C, output_dtype_str, warps_m, ctas_per_row, bytes_per_ldg, kernel_cfg, occupancy
    )

    ws_size = _compute_rmsnorm_silu_workspace_size(
        num_tokens,
        C,
        output_dtype_str,
        warps_m,
        ctas_per_row,
        kernel_cfg,
        occupancy,
        sm_count,
    )
    workspace = torch.empty(ws_size, dtype=torch.uint8, device=input.device)

    if output_dtype_str == "nvfp4":
        num_blocks = C // 16
        if block_scale is None:
            block_scale = torch.empty(
                num_tokens, num_blocks, dtype=torch.float8_e4m3fn, device=input.device
            )
        else:
            expected_shape = (num_tokens, num_blocks)
            if tuple(block_scale.shape) != expected_shape:
                raise ValueError(
                    f"block_scale shape mismatch: expected {expected_shape}, "
                    f"got {tuple(block_scale.shape)}"
                )
            if block_scale.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"block_scale must be float8_e4m3fn, got {block_scale.dtype}"
                )
        scale_row_out = block_scale
    else:
        scale_row_out = torch.empty(0, dtype=torch.uint8, device=input.device)

    module.rmsnorm_silu(out, input, weight, eps, workspace, scale_row_out, sm_count)

    if output_dtype_str == "nvfp4":
        return out, block_scale

    return out


####################################################################################################
# Fused QK RMSNorm + 3D RoPE for Video Generation DIT Self-Attention
####################################################################################################


@supported_compute_capability([80, 86, 89, 90, 100, 103, 110, 120, 121])
def _check_fused_qk_rmsnorm_rope(
    qkv,
    q_weight,
    k_weight,
    **kwargs,
):
    """Validate inputs for fused QK RMSNorm + 3D RoPE.

    Architecture notes:
    - SM80+ (Ampere): Full support for BF16 path; FP8 output uses software emulation
    - SM89+ (Ada): Native FP8 E4M3 conversion instructions (faster FP8 output)
    - SM90 (Hopper): Primary target architecture
    - SM100/103 (Blackwell B200, B300): Native float2 packed math (FFMA2); primary target
    All SM100+/SM89+ features have scalar fallbacks, so SM80 is the true minimum.
    """
    if not qkv.is_cuda:
        raise ValueError("qkv must be a CUDA tensor")
    if qkv.dtype != torch.bfloat16:
        raise ValueError("qkv must be bfloat16")
    if not qkv.is_contiguous():
        raise ValueError("qkv must be contiguous")
    if qkv.ndim not in (2, 3):
        raise ValueError(
            f"qkv must be 2D [num_tokens, hidden] or 3D [batch, seq_len, hidden], "
            f"got {qkv.ndim}D"
        )

    head_dim = kwargs.get("head_dim")
    if head_dim not in (64, 128, 256):
        raise ValueError(f"head_dim must be 64, 128, or 256, got {head_dim}")

    num_heads_q = kwargs.get("num_heads_q")
    num_heads_k = kwargs.get("num_heads_k")
    num_heads_v = kwargs.get("num_heads_v")
    max_heads = max(num_heads_q, num_heads_k, num_heads_v)
    if max_heads > 32:
        raise ValueError(
            f"max(num_heads_q, num_heads_k, num_heads_v) must be <= 32, got {max_heads}"
        )

    num_frame_channels = kwargs.get("num_frame_channels")
    num_height_channels = kwargs.get("num_height_channels")
    num_width_channels = kwargs.get("num_width_channels")
    if num_frame_channels + num_height_channels + num_width_channels != head_dim:
        raise ValueError(
            f"num_frame_channels ({num_frame_channels}) + num_height_channels "
            f"({num_height_channels}) + num_width_channels ({num_width_channels}) "
            f"must equal head_dim ({head_dim})"
        )
    if (
        num_frame_channels % 2 != 0
        or num_height_channels % 2 != 0
        or num_width_channels % 2 != 0
    ):
        raise ValueError(
            f"Channel counts must all be even (freq table uses count/2), got "
            f"frame={num_frame_channels}, height={num_height_channels}, "
            f"width={num_width_channels}"
        )

    ppf = kwargs.get("ppf")
    pph = kwargs.get("pph")
    ppw = kwargs.get("ppw")
    if ppf <= 0 or pph <= 0 or ppw <= 0:
        raise ValueError(f"ppf, pph, ppw must be positive, got ({ppf}, {pph}, {ppw})")
    expected_seq_len = ppf * pph * ppw
    if qkv.ndim == 3:
        actual_seq_len = qkv.shape[1]
        if actual_seq_len != expected_seq_len:
            raise ValueError(
                f"qkv seq_len ({actual_seq_len}) != ppf*pph*ppw ({expected_seq_len})"
            )
    else:
        num_tokens = qkv.shape[0]
        if num_tokens % expected_seq_len != 0:
            raise ValueError(
                f"qkv num_tokens ({num_tokens}) must be divisible by "
                f"ppf*pph*ppw ({expected_seq_len})"
            )

    return True


@flashinfer_api
@backend_requirement(backend_checks={}, common_check=_check_fused_qk_rmsnorm_rope)
def fused_qk_rmsnorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    ppf: int,
    pph: int,
    ppw: int,
    num_frame_channels: int,
    num_height_channels: int,
    num_width_channels: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float = 1e-6,
    base: float = 10000.0,
    interleave: bool = True,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    is_qk_norm: bool = True,
    output_fp8: bool = False,
    output_quant_scale: float = 1.0,
    v_quant_scale: float = 1.0,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Fused QK RMSNorm + 3D RoPE + V copy for video generation DIT self-attention.

    Applies across-heads RMSNorm to Q and K, then rotary position embeddings
    with 3D spatial decomposition (frame/height/width), and copies V to a
    contiguous output buffer. Optionally quantizes all outputs to FP8 E4M3.

    Parameters
    ----------
    qkv : torch.Tensor
        Combined QKV input, BF16, contiguous. Accepted shapes:
        - 3D: ``[batch, seq_len, (num_heads_q+num_heads_k+num_heads_v)*head_dim]``
        - 2D: ``[num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]``
          where ``num_tokens`` must be divisible by ``ppf*pph*ppw``.
    q_weight : torch.Tensor
        RMSNorm weight for Q ``[num_heads_q * head_dim]``, BF16.
    k_weight : torch.Tensor
        RMSNorm weight for K ``[num_heads_k * head_dim]``, BF16.
    ppf : int
        Number of patches in frame dimension.
    pph : int
        Number of patches in height dimension.
    ppw : int
        Number of patches in width dimension.
        ``seq_len = ppf * pph * ppw``.
    num_frame_channels : int
        RoPE frequency channels for the frame dimension (must be even).
    num_height_channels : int
        RoPE frequency channels for the height dimension (must be even).
    num_width_channels : int
        RoPE frequency channels for the width dimension (must be even).
        ``num_frame_channels + num_height_channels + num_width_channels == head_dim``.
    num_heads_q : int
        Number of query heads.
    num_heads_k : int
        Number of key heads.
    num_heads_v : int
        Number of value heads.
    head_dim : int
        Dimension per head (must be 64, 128, or 256).
    eps : float
        RMSNorm epsilon.
    base : float
        RoPE base frequency.
    interleave : bool
        True for interleaved RoPE (non-NeoX style), False for NeoX-style.
    factor : float
        YARN RoPE scaling factor. 1.0 disables YARN.
    low : float
        YARN low frequency threshold.
    high : float
        YARN high frequency threshold.
    attention_factor : float
        YARN attention factor applied to cos/sin. Must be 1.0 when factor is 1.0.
    is_qk_norm : bool
        Whether to apply RMSNorm (False = RoPE only, skip normalization).
    output_fp8 : bool
        Quantize Q, K, V outputs to FP8 E4M3.
    output_quant_scale : float
        FP8 quantization scale for Q and K outputs.
    v_quant_scale : float
        FP8 quantization scale for V output.
    q_out : Optional[torch.Tensor]
        Pre-allocated Q output tensor (destination-passing style).
    k_out : Optional[torch.Tensor]
        Pre-allocated K output tensor.
    v_out : Optional[torch.Tensor]
        Pre-allocated V output tensor.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(q_out, k_out, v_out)``. If input is 3D, each has shape
        ``[batch, seq_len, num_heads_x, head_dim]``. If input is 2D,
        each has shape ``[num_tokens, num_heads_x, head_dim]``.
    """
    out_dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
    seq_len = ppf * pph * ppw

    out_shape_q: tuple[int, ...]
    out_shape_k: tuple[int, ...]
    out_shape_v: tuple[int, ...]
    if qkv.ndim == 3:
        batch_size = qkv.shape[0]
        num_tokens = batch_size * seq_len
        out_shape_q = (batch_size, seq_len, num_heads_q, head_dim)
        out_shape_k = (batch_size, seq_len, num_heads_k, head_dim)
        out_shape_v = (batch_size, seq_len, num_heads_v, head_dim)
    else:
        num_tokens = qkv.shape[0]
        out_shape_q = (num_tokens, num_heads_q, head_dim)
        out_shape_k = (num_tokens, num_heads_k, head_dim)
        out_shape_v = (num_tokens, num_heads_v, head_dim)

    # Validate weights
    expected_q_weight_numel = num_heads_q * head_dim
    expected_k_weight_numel = num_heads_k * head_dim
    if q_weight.numel() != expected_q_weight_numel:
        raise ValueError(
            f"q_weight size {q_weight.numel()} != num_heads_q*head_dim ({expected_q_weight_numel})"
        )
    if k_weight.numel() != expected_k_weight_numel:
        raise ValueError(
            f"k_weight size {k_weight.numel()} != num_heads_k*head_dim ({expected_k_weight_numel})"
        )
    if q_weight.dtype != torch.bfloat16 or k_weight.dtype != torch.bfloat16:
        raise ValueError("q_weight and k_weight must be bfloat16")
    if not q_weight.is_contiguous() or not k_weight.is_contiguous():
        raise ValueError("q_weight and k_weight must be contiguous")

    if q_out is None:
        q_out = torch.empty(*out_shape_q, dtype=out_dtype, device=qkv.device)
    if k_out is None:
        k_out = torch.empty(*out_shape_k, dtype=out_dtype, device=qkv.device)
    if v_out is None:
        v_out = torch.empty(*out_shape_v, dtype=out_dtype, device=qkv.device)

    # Validate user-supplied output buffers
    for name, buf, expected_shape in [
        ("q_out", q_out, out_shape_q),
        ("k_out", k_out, out_shape_k),
        ("v_out", v_out, out_shape_v),
    ]:
        if tuple(buf.shape) != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(buf.shape)} != expected {expected_shape}"
            )
        if buf.dtype != out_dtype:
            raise ValueError(f"{name} dtype {buf.dtype} != expected {out_dtype}")
        if not buf.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if buf.device != qkv.device:
            raise ValueError(f"{name} device {buf.device} != qkv device {qkv.device}")

    qkv_flat = qkv.view(num_tokens, -1)
    q_out_flat = q_out.view(num_tokens, -1)
    k_out_flat = k_out.view(num_tokens, -1)
    v_out_flat = v_out.view(num_tokens, -1)

    get_norm_module().fused_qk_rmsnorm_rope(
        qkv_flat,
        q_weight,
        k_weight,
        q_out_flat,
        k_out_flat,
        v_out_flat,
        num_tokens,
        seq_len,
        ppf,
        pph,
        ppw,
        num_frame_channels,
        num_height_channels,
        num_width_channels,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        float(eps),
        float(base),
        interleave,
        float(factor),
        float(low),
        float(high),
        float(attention_factor),
        is_qk_norm,
        output_fp8,
        float(output_quant_scale),
        float(v_quant_scale),
    )

    return q_out, k_out, v_out


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
    "fused_qk_rmsnorm_rope",
]
