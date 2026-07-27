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
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..quantization.fp4_quantization import nvfp4_quantize
from ..tllm_enums import SfLayout
from ..trace.templates.conv import conv3d_nvfp4_trace
from ..utils import supported_compute_capability

_NVFP4_BLOCK_SIZE = 16
_NVFP4_GLOBAL_QUANT_MAX = 448.0 * 6.0
_SUPPORTED_PADDING = ((0, 0, 0), (0, 1, 1))


def _normalize_triple(
    value: int | Tuple[int, int, int], name: str
) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(
            f"{name} must be an int or a tuple of three ints; got {value!r}"
        )
    if not all(isinstance(item, int) for item in value):
        raise TypeError(f"{name} must contain ints; got {value!r}")
    return value


def _check_cuda_tensor(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor; got {tensor.device}")


def _check_global_scale(
    scale: torch.Tensor,
    *,
    device: torch.device,
    name: str,
) -> None:
    _check_cuda_tensor(scale, name)
    if scale.device != device:
        raise ValueError(f"{name} must be on {device}; got {scale.device}")
    if scale.dtype != torch.float32:
        raise TypeError(f"{name} must have float32 dtype; got {scale.dtype}")
    if tuple(scale.shape) != (1,):
        raise ValueError(f"{name} must have shape (1,); got {tuple(scale.shape)}")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_logical_weight(weight: torch.Tensor) -> Tuple[int, int, int, int, int]:
    _check_cuda_tensor(weight, "weight")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"weight must have bfloat16 dtype; got {weight.dtype}")
    if weight.dim() != 5:
        raise ValueError(
            "weight must use logical KCTRS layout with rank five; "
            f"got shape {tuple(weight.shape)}"
        )
    out_channels, in_channels, filter_t, filter_r, filter_s = map(int, weight.shape)
    if (filter_t, filter_r, filter_s) != (3, 3, 3):
        raise ValueError(
            "SM120 NVFP4 Conv3d currently supports only 3x3x3 weights; "
            f"got {(filter_t, filter_r, filter_s)}"
        )
    if in_channels % 128 != 0:
        raise ValueError(
            f"weight input channels must be a multiple of 128; got {in_channels}"
        )
    if out_channels % 128 != 0:
        raise ValueError(
            f"weight output channels must be a multiple of 128; got {out_channels}"
        )
    return out_channels, in_channels, filter_t, filter_r, filter_s


@flashinfer_api
def prepare_nvfp4_conv3d_weight(
    weight: torch.Tensor,
    weight_global_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a BF16 KCTRS weight for the SM120 NVFP4 Conv3d operator.

    Preparation is an offline/model-load operation. The logical KCTRS weight is
    reordered to KTRSC, quantized in contiguous C16 groups, and returned in the
    packed E2M1 plus 128x4-swizzled E4M3 representation consumed by the
    block-scaled Conv3d kernel.

    Parameters
    ----------
    weight:
        CUDA BF16 tensor in logical ``(K, C, T, R, S)`` layout. PR1 supports
        ``(T, R, S) == (3, 3, 3)`` and C/K multiples of 128.
    weight_global_scale:
        Optional CUDA float32 tensor with shape ``(1,)`` containing the global
        quantization multiplier. If omitted, it is derived from the weight
        absolute maximum.

    Returns
    -------
    packed_weight:
        CUDA uint8 tensor with shape ``(K, T, R, S, C // 2)``. The low nibble
        stores the even C element and the high nibble stores the odd C element.
    weight_scale:
        CUDA uint8 buffer containing E4M3 C16 scales in FlashInfer's canonical
        128x4 block-scaled layout.
    weight_global_scale:
        CUDA float32 tensor with shape ``(1,)``. Pass this tensor unchanged to
        :func:`conv3d_nvfp4`.
    """

    out_channels, in_channels, filter_t, filter_r, filter_s = _check_logical_weight(
        weight
    )
    if weight_global_scale is None:
        weight_global_scale = (
            _NVFP4_GLOBAL_QUANT_MAX
            / weight.detach().abs().amax().float().clamp(min=1e-8)
        ).reshape(1)
    else:
        _check_global_scale(
            weight_global_scale,
            device=weight.device,
            name="weight_global_scale",
        )

    flattened_k = in_channels * filter_t * filter_r * filter_s
    weight_matrix = (
        weight.permute(0, 2, 3, 4, 1).contiguous().reshape(out_channels, flattened_k)
    )
    packed_weight, weight_scale = nvfp4_quantize(
        weight_matrix,
        weight_global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    packed_weight = packed_weight.reshape(
        out_channels,
        filter_t,
        filter_r,
        filter_s,
        in_channels // 2,
    )
    return (
        packed_weight,
        weight_scale.contiguous(),
        weight_global_scale.contiguous(),
    )


@lru_cache(maxsize=1)
def _activation_quantization_module():
    from ..jit.conv3d import gen_conv3d_nvfp4_activation_module

    return gen_conv3d_nvfp4_activation_module().build_and_load()


def _quantize_nvfp4_conv3d_activation(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
    padding: Tuple[int, int, int],
    *,
    tile_variant: int = 0,
    packed_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_cuda_tensor(input, "input")
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must have bfloat16 dtype; got {input.dtype}")
    if input.dim() != 5:
        raise ValueError(
            f"input must use logical NCDHW layout with rank five; got {tuple(input.shape)}"
        )
    if not input.is_contiguous():
        raise ValueError("input must be contiguous in NCDHW layout")
    _check_global_scale(
        input_global_scale,
        device=input.device,
        name="input_global_scale",
    )
    if padding not in _SUPPORTED_PADDING:
        raise ValueError(f"padding must be one of {_SUPPORTED_PADDING}; got {padding}")
    if tile_variant not in range(5):
        raise ValueError(f"tile_variant must be in [0, 4]; got {tile_variant}")

    batch, channels, depth, height, width = map(int, input.shape)
    if channels % 128 != 0:
        raise ValueError(f"input channels must be a multiple of 128; got {channels}")
    pad_depth, pad_height, pad_width = padding
    if pad_depth != 0:
        raise ValueError("depth halo materialization is not supported")
    physical_height = height + 2 * pad_height
    physical_width = width + 2 * pad_width
    packed_shape = (
        batch,
        depth,
        physical_height,
        physical_width,
        channels // 2,
    )
    scale_shape = (
        batch,
        depth,
        physical_height,
        physical_width,
        channels // _NVFP4_BLOCK_SIZE,
    )
    if packed_out is None:
        packed = torch.empty(
            packed_shape,
            dtype=torch.uint8,
            device=input.device,
        )
    else:
        _check_cuda_tensor(packed_out, "packed_out")
        if packed_out.device != input.device:
            raise ValueError(
                f"packed_out must be on {input.device}; got {packed_out.device}"
            )
        if packed_out.dtype != torch.uint8:
            raise TypeError(f"packed_out must have uint8 dtype; got {packed_out.dtype}")
        if tuple(packed_out.shape) != packed_shape:
            raise ValueError(
                f"packed_out must have shape {packed_shape}; "
                f"got {tuple(packed_out.shape)}"
            )
        if not packed_out.is_contiguous():
            raise ValueError("packed_out must be contiguous")
        packed = packed_out
    if scale_out is None:
        scales = torch.empty(
            scale_shape,
            dtype=torch.uint8,
            device=input.device,
        )
    else:
        _check_cuda_tensor(scale_out, "scale_out")
        if scale_out.device != input.device:
            raise ValueError(
                f"scale_out must be on {input.device}; got {scale_out.device}"
            )
        if scale_out.dtype != torch.uint8:
            raise TypeError(f"scale_out must have uint8 dtype; got {scale_out.dtype}")
        if tuple(scale_out.shape) != scale_shape:
            raise ValueError(
                f"scale_out must have shape {scale_shape}; got {tuple(scale_out.shape)}"
            )
        if not scale_out.is_contiguous():
            raise ValueError("scale_out must be contiguous")
        scales = scale_out
    _activation_quantization_module().nvfp4_conv3d_quantize_activation(
        input,
        input_global_scale,
        packed,
        scales,
        pad_height,
        pad_width,
        tile_variant,
    )
    return packed, scales


def _check_packed_weight(
    packed_weight: torch.Tensor,
    *,
    input_channels: int,
    device: torch.device,
) -> int:
    _check_cuda_tensor(packed_weight, "packed_weight")
    if packed_weight.device != device:
        raise ValueError(
            f"packed_weight must be on {device}; got {packed_weight.device}"
        )
    if packed_weight.dtype != torch.uint8:
        raise TypeError(
            f"packed_weight must have uint8 dtype; got {packed_weight.dtype}"
        )
    if packed_weight.dim() != 5:
        raise ValueError(
            "packed_weight must have shape (K, 3, 3, 3, C // 2); "
            f"got {tuple(packed_weight.shape)}"
        )
    output_channels, filter_t, filter_r, filter_s, packed_channels = map(
        int, packed_weight.shape
    )
    expected_shape = (output_channels, 3, 3, 3, input_channels // 2)
    if tuple(packed_weight.shape) != expected_shape:
        raise ValueError(
            "packed_weight must have shape (K, 3, 3, 3, C // 2); "
            f"expected {expected_shape}, got {tuple(packed_weight.shape)}"
        )
    if output_channels % 128 != 0:
        raise ValueError(
            f"packed_weight output channels must be a multiple of 128; got {output_channels}"
        )
    if (filter_t, filter_r, filter_s) != (
        3,
        3,
        3,
    ) or packed_channels * 2 != input_channels:
        raise ValueError(
            "packed_weight must have shape (K, 3, 3, 3, C // 2); "
            f"got {tuple(packed_weight.shape)}"
        )
    if not packed_weight.is_contiguous():
        raise ValueError("packed_weight must be contiguous")
    return output_channels


def _required_weight_scale_bytes(output_channels: int, input_channels: int) -> int:
    flattened_k = input_channels * 3 * 3 * 3
    return (
        ((output_channels + 127) // 128)
        * (((flattened_k // _NVFP4_BLOCK_SIZE) + 3) // 4)
        * 512
    )


def _check_weight_scale(
    weight_scale: torch.Tensor,
    *,
    output_channels: int,
    input_channels: int,
    device: torch.device,
) -> None:
    _check_cuda_tensor(weight_scale, "weight_scale")
    if weight_scale.device != device:
        raise ValueError(f"weight_scale must be on {device}; got {weight_scale.device}")
    if weight_scale.dtype != torch.uint8:
        raise TypeError(f"weight_scale must have uint8 dtype; got {weight_scale.dtype}")
    if not weight_scale.is_contiguous():
        raise ValueError("weight_scale must be contiguous")
    required_bytes = _required_weight_scale_bytes(output_channels, input_channels)
    if weight_scale.numel() != required_bytes:
        raise ValueError(
            "weight_scale has the wrong canonical 128x4 buffer size; "
            f"expected {required_bytes} bytes, got {weight_scale.numel()}"
        )


def _check_bias(
    bias: torch.Tensor,
    *,
    output_channels: int,
    device: torch.device,
) -> None:
    _check_cuda_tensor(bias, "bias")
    if bias.device != device:
        raise ValueError(f"bias must be on {device}; got {bias.device}")
    if bias.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"bias must have bfloat16 or float32 dtype; got {bias.dtype}")
    if tuple(bias.shape) != (output_channels,):
        raise ValueError(
            f"bias must have shape ({output_channels},); got {tuple(bias.shape)}"
        )
    if not bias.is_contiguous():
        raise ValueError("bias must be contiguous")


def _check_output(
    output: torch.Tensor,
    *,
    expected_shape: Tuple[int, int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    _check_cuda_tensor(output, "out")
    if output.device != device:
        raise ValueError(f"out must be on {device}; got {output.device}")
    if output.dtype != torch.bfloat16:
        raise TypeError(f"out must have bfloat16 dtype; got {output.dtype}")
    if tuple(output.shape) != expected_shape:
        raise ValueError(
            f"out must have shape {expected_shape}; got {tuple(output.shape)}"
        )
    if not output.is_contiguous(memory_format=torch.channels_last_3d):
        raise ValueError("out must be contiguous in channels_last_3d memory format")
    return output.permute(0, 2, 3, 4, 1)


@torch.library.custom_op(
    "flashinfer::conv3d_nvfp4",
    mutates_args=("out",),
    device_types="cuda",
)
def _conv3d_nvfp4_custom_op(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    pad_height: int,
    pad_width: int,
) -> None:
    packed_input, input_scale = _quantize_nvfp4_conv3d_activation(
        input,
        input_global_scale,
        (0, pad_height, pad_width),
    )
    alpha = torch.reciprocal(input_global_scale * weight_global_scale)
    if bias is None:
        alpha_and_bias = alpha
    else:
        alpha_and_bias = torch.cat(
            (
                alpha,
                torch.zeros_like(alpha),
                bias.to(dtype=torch.float32),
            )
        )

    from .nvfp4_sm120 import run_sm120_nvfp4_conv3d

    run_sm120_nvfp4_conv3d(
        packed_input,
        packed_weight,
        input_scale,
        weight_scale,
        alpha_and_bias,
        out.permute(0, 2, 3, 4, 1),
        fuse_bias=bias is not None,
    )


@_conv3d_nvfp4_custom_op.register_fake
def _conv3d_nvfp4_fake(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    pad_height: int,
    pad_width: int,
) -> None:
    return None


@supported_compute_capability([120])
@flashinfer_api(trace=conv3d_nvfp4_trace)
def conv3d_nvfp4(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    stride: int | Tuple[int, int, int] = 1,
    padding: int | Tuple[int, int, int] = (0, 1, 1),
    dilation: int | Tuple[int, int, int] = 1,
    groups: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the SM120 W4A4 block-scaled 3x3x3 Conv3d kernel.

    ``input`` uses logical contiguous NCDHW BF16 layout. ``packed_weight`` and
    ``weight_scale`` must be returned by :func:`prepare_nvfp4_conv3d_weight`.
    PR1 supports batch one, C/K multiples of 128, unit stride and dilation,
    one group, and either no padding or H/W padding one. The returned logical
    NCDHW tensor is physically contiguous in channels-last-3d format.
    """

    _check_cuda_tensor(input, "input")
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must have bfloat16 dtype; got {input.dtype}")
    if input.dim() != 5:
        raise ValueError(
            f"input must use logical NCDHW layout with rank five; got {tuple(input.shape)}"
        )
    if not input.is_contiguous():
        raise ValueError("input must be contiguous in NCDHW layout")
    if torch.cuda.get_device_capability(input.device) != (12, 0):
        raise RuntimeError(
            "conv3d_nvfp4 requires an SM120 GPU; "
            f"got compute capability {torch.cuda.get_device_capability(input.device)}"
        )
    if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
        raise RuntimeError(
            f"conv3d_nvfp4 requires CUDA 13 or newer; got {torch.version.cuda}"
        )

    stride = _normalize_triple(stride, "stride")
    padding = _normalize_triple(padding, "padding")
    dilation = _normalize_triple(dilation, "dilation")
    if stride != (1, 1, 1):
        raise ValueError(f"stride must be (1, 1, 1); got {stride}")
    if padding not in _SUPPORTED_PADDING:
        raise ValueError(f"padding must be one of {_SUPPORTED_PADDING}; got {padding}")
    if dilation != (1, 1, 1):
        raise ValueError(f"dilation must be (1, 1, 1); got {dilation}")
    if groups != 1:
        raise ValueError(f"groups must be 1; got {groups}")

    batch, input_channels, input_depth, input_height, input_width = map(
        int, input.shape
    )
    if batch != 1:
        raise ValueError(f"input batch must be 1; got {batch}")
    if input_channels % 128 != 0:
        raise ValueError(
            f"input channels must be a multiple of 128; got {input_channels}"
        )
    output_depth = input_depth - 2
    output_height = input_height + 2 * padding[1] - 2
    output_width = input_width + 2 * padding[2] - 2
    if min(output_depth, output_height, output_width) <= 0:
        raise ValueError(
            "input spatial dimensions are too small for a 3x3x3 kernel; "
            f"got {(input_depth, input_height, input_width)} with padding {padding}"
        )
    if output_depth * output_height * output_width < 2:
        raise ValueError(
            "SM120 NVFP4 Conv3d requires at least two output spatial positions; "
            f"got output shape {(output_depth, output_height, output_width)}"
        )

    output_channels = _check_packed_weight(
        packed_weight,
        input_channels=input_channels,
        device=input.device,
    )
    _check_weight_scale(
        weight_scale,
        output_channels=output_channels,
        input_channels=input_channels,
        device=input.device,
    )
    _check_global_scale(
        input_global_scale,
        device=input.device,
        name="input_global_scale",
    )
    _check_global_scale(
        weight_global_scale,
        device=input.device,
        name="weight_global_scale",
    )
    if bias is not None:
        _check_bias(
            bias,
            output_channels=output_channels,
            device=input.device,
        )

    expected_shape = (
        batch,
        output_channels,
        output_depth,
        output_height,
        output_width,
    )
    if out is None:
        out = torch.empty(
            expected_shape,
            dtype=torch.bfloat16,
            device=input.device,
            memory_format=torch.channels_last_3d,
        )
    else:
        _check_output(
            out,
            expected_shape=expected_shape,
            device=input.device,
        )

    _conv3d_nvfp4_custom_op(
        input,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        bias,
        out,
        padding[1],
        padding[2],
    )
    return out


__all__ = ["conv3d_nvfp4", "prepare_nvfp4_conv3d_weight"]
