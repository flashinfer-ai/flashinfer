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
from typing import Optional, Union

import torch

from ..api_logging import flashinfer_api
from ..utils import (
    device_support_pdl,
    get_compute_capability,
    register_custom_op,
    register_fake_op,
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

if _USE_CUDA_NORM:

    @functools.cache
    def get_norm_module():
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


@flashinfer_api
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


@flashinfer_api
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


@flashinfer_api
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
