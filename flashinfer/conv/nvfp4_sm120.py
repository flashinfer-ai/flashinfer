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

from functools import cache
from typing import Tuple

import torch


def _kernel_source_files() -> tuple[str, ...]:
    from .kernels import (
        _sm120_blockscaled_dispatch,
        _sm120_blockscaled_gemm,
        _sm120_conv3d_descriptor,
        _sm120_conv3d_mainloop,
        conv3d_nvfp4_sm120,
    )

    return (
        __file__,
        _sm120_blockscaled_dispatch.__file__,
        _sm120_blockscaled_gemm.__file__,
        _sm120_conv3d_descriptor.__file__,
        _sm120_conv3d_mainloop.__file__,
        conv3d_nvfp4_sm120.__file__,
    )


def _kernel_name(
    input_shape: Tuple[int, int, int, int, int],
    output_channels: int,
    *,
    fuse_alpha: bool,
    fuse_bias: bool,
    a_copy_bits: int,
    a_copy_layout: str,
    a_producer_warps: int,
    n_pair: bool,
    swizzle_size: int,
) -> str:
    n, c, d, h, w = input_shape
    return (
        f"n{n}_c{c}_d{d}_h{h}_w{w}_k{output_channels}"
        f"_alpha{int(fuse_alpha)}_b{int(fuse_bias)}"
        f"_a{a_copy_bits}{a_copy_layout}"
        f"_aw{a_producer_warps}_np{int(n_pair)}_sw{swizzle_size}"
    )


def _weight_scale_num_bytes(output_channels: int, input_channels: int) -> int:
    return (
        ((output_channels + 127) // 128)
        * (((input_channels * 27 // 16) + 3) // 4)
        * 512
    )


@cache
def _get_compiled_kernel(
    input_shape: Tuple[int, int, int, int, int],
    output_channels: int,
    *,
    fuse_alpha: bool,
    fuse_bias: bool,
    a_copy_bits: int = 128,
    a_copy_layout: str = "coalesced",
    a_producer_warps: int = 4,
    n_pair: bool = True,
    swizzle_size: int = 2,
    device_index: int,
):
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl.utils import get_max_active_clusters
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from .kernels.conv3d_nvfp4_sm120 import Sm120Nvfp4Conv3dKernel

    batch, channels, physical_depth, physical_height, physical_width = input_shape
    filter_t = filter_r = filter_s = 3
    output_depth = physical_depth - filter_t + 1
    output_height = physical_height - filter_r + 1
    output_width = physical_width - filter_s + 1
    if min(output_depth, output_height, output_width) <= 0:
        raise ValueError(
            "physical input must be at least 3 in every spatial dimension; "
            f"got {input_shape}"
        )

    kernel = Sm120Nvfp4Conv3dKernel(
        a_copy_bits=a_copy_bits,
        a_copy_layout=a_copy_layout,
        a_producer_warps=a_producer_warps,
        n_pair=n_pair,
        fuse_alpha=fuse_alpha,
        fuse_bias=fuse_bias,
        raster_order="n",
        swizzle_size=swizzle_size,
    )
    with torch.cuda.device(device_index):
        max_active_clusters = get_max_active_clusters(1)
    weight_scale_bytes = _weight_scale_num_bytes(output_channels, channels)

    def compile_kernel():
        packed_input = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (
                batch,
                physical_depth,
                physical_height,
                physical_width,
                channels // 2,
            ),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        packed_weight = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (
                output_channels,
                filter_t,
                filter_r,
                filter_s,
                channels // 2,
            ),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        input_scale = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (
                batch,
                physical_depth,
                physical_height,
                physical_width,
                channels // 16,
            ),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        weight_scale = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (weight_scale_bytes,),
            assumed_align=16,
        )
        alpha_and_bias = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (output_channels + 2 if fuse_bias else 1,),
            assumed_align=4,
        )
        output = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (
                batch,
                output_depth,
                output_height,
                output_width,
                output_channels,
            ),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel.wrapper,
            packed_input,
            packed_weight,
            input_scale,
            weight_scale,
            alpha_and_bias,
            output,
            max_active_clusters,
            stream,
            options="--opt-level 3 --enable-tvm-ffi",
        )

    name = _kernel_name(
        input_shape,
        output_channels,
        fuse_alpha=fuse_alpha,
        fuse_bias=fuse_bias,
        a_copy_bits=a_copy_bits,
        a_copy_layout=a_copy_layout,
        a_producer_warps=a_producer_warps,
        n_pair=n_pair,
        swizzle_size=swizzle_size,
    )
    return build_and_load_cute_dsl_kernel(
        "conv3d_nvfp4",
        name,
        compile_kernel,
        extra_key_files=_kernel_source_files(),
    )


def _check_runtime_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    ndim: int,
    alignment: int,
) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor; got {tensor.device}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}; got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have {dtype} dtype; got {tensor.dtype}")
    if tensor.dim() != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}; got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % alignment != 0:
        raise ValueError(f"{name} must be {alignment}-byte aligned")


def _validate_runtime_tensors(
    packed_input: torch.Tensor,
    packed_weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha_and_bias: torch.Tensor,
    output: torch.Tensor,
    *,
    fuse_alpha: bool,
    fuse_bias: bool,
) -> int:
    if not packed_input.is_cuda:
        raise ValueError(
            f"packed_input must be a CUDA tensor; got {packed_input.device}"
        )
    device = packed_input.device
    _check_runtime_tensor(
        packed_input,
        name="packed_input",
        dtype=torch.uint8,
        device=device,
        ndim=5,
        alignment=16,
    )
    _check_runtime_tensor(
        packed_weight,
        name="packed_weight",
        dtype=torch.uint8,
        device=device,
        ndim=5,
        alignment=16,
    )
    _check_runtime_tensor(
        input_scale,
        name="input_scale",
        dtype=torch.uint8,
        device=device,
        ndim=5,
        alignment=16,
    )
    _check_runtime_tensor(
        weight_scale,
        name="weight_scale",
        dtype=torch.uint8,
        device=device,
        ndim=1,
        alignment=16,
    )
    _check_runtime_tensor(
        alpha_and_bias,
        name="alpha_and_bias",
        dtype=torch.float32,
        device=device,
        ndim=1,
        alignment=4,
    )
    _check_runtime_tensor(
        output,
        name="output",
        dtype=torch.bfloat16,
        device=device,
        ndim=5,
        alignment=16,
    )

    batch, physical_depth, physical_height, physical_width, packed_channels = map(
        int, packed_input.shape
    )
    channels = packed_channels * 2
    output_channels = int(packed_weight.shape[0])
    if batch != 1:
        raise ValueError(f"packed_input batch must be 1; got {batch}")
    if channels % 128 != 0:
        raise ValueError(f"input channels must be a multiple of 128; got {channels}")
    if output_channels % 128 != 0:
        raise ValueError(
            f"output channels must be a multiple of 128; got {output_channels}"
        )
    expected_weight_shape = (output_channels, 3, 3, 3, packed_channels)
    if tuple(packed_weight.shape) != expected_weight_shape:
        raise ValueError(
            f"packed_weight must have shape {expected_weight_shape}; "
            f"got {tuple(packed_weight.shape)}"
        )
    expected_scale_shape = (
        batch,
        physical_depth,
        physical_height,
        physical_width,
        channels // 16,
    )
    if tuple(input_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"input_scale must have shape {expected_scale_shape}; "
            f"got {tuple(input_scale.shape)}"
        )
    expected_output_shape = (
        batch,
        physical_depth - 2,
        physical_height - 2,
        physical_width - 2,
        output_channels,
    )
    if min(expected_output_shape[1:4]) <= 0:
        raise ValueError(
            "packed_input spatial dimensions must be at least 3; "
            f"got {tuple(packed_input.shape)}"
        )
    if tuple(output.shape) != expected_output_shape:
        raise ValueError(
            f"output must be contiguous NDHWC with shape {expected_output_shape}; "
            f"got {tuple(output.shape)}"
        )
    expected_weight_scale_bytes = _weight_scale_num_bytes(output_channels, channels)
    if weight_scale.numel() != expected_weight_scale_bytes:
        raise ValueError(
            f"weight_scale must contain {expected_weight_scale_bytes} bytes; "
            f"got {weight_scale.numel()}"
        )
    expected_alpha_and_bias = output_channels + 2 if fuse_bias else 1
    if alpha_and_bias.numel() != expected_alpha_and_bias:
        raise ValueError(
            f"alpha_and_bias must contain {expected_alpha_and_bias} values; "
            f"got {alpha_and_bias.numel()}"
        )
    if fuse_bias and not fuse_alpha:
        raise ValueError("fuse_bias requires fuse_alpha")

    device_index = device.index
    if device_index is None:
        raise ValueError("packed_input must have a concrete CUDA device index")
    return device_index


def run_sm120_nvfp4_conv3d(
    packed_input: torch.Tensor,
    packed_weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha_and_bias: torch.Tensor,
    output: torch.Tensor,
    *,
    fuse_alpha: bool = True,
    fuse_bias: bool,
    a_copy_bits: int = 128,
    a_copy_layout: str = "coalesced",
    a_producer_warps: int = 4,
    n_pair: bool = True,
    swizzle_size: int = 2,
) -> None:
    if not weight_scale.is_contiguous():
        raise ValueError("weight_scale must be contiguous")
    weight_scale = weight_scale.reshape(-1)
    device_index = _validate_runtime_tensors(
        packed_input,
        packed_weight,
        input_scale,
        weight_scale,
        alpha_and_bias,
        output,
        fuse_alpha=fuse_alpha,
        fuse_bias=fuse_bias,
    )
    input_shape = (
        int(packed_input.shape[0]),
        int(packed_input.shape[4]) * 2,
        int(packed_input.shape[1]),
        int(packed_input.shape[2]),
        int(packed_input.shape[3]),
    )
    output_channels = int(packed_weight.shape[0])
    if output_channels % 256 != 0:
        n_pair = False
    output_m = (
        input_shape[0]
        * (input_shape[2] - 2)
        * (input_shape[3] - 2)
        * (input_shape[4] - 2)
    )
    m_tiles = (output_m + 127) // 128
    swizzle_size = max(1, swizzle_size)
    while swizzle_size > max(1, m_tiles):
        swizzle_size //= 2
    compiled = _get_compiled_kernel(
        input_shape,
        output_channels,
        fuse_alpha=fuse_alpha,
        fuse_bias=fuse_bias,
        a_copy_bits=a_copy_bits,
        a_copy_layout=a_copy_layout,
        a_producer_warps=a_producer_warps,
        n_pair=n_pair,
        swizzle_size=swizzle_size,
        device_index=device_index,
    )
    compiled(
        packed_input,
        packed_weight,
        input_scale,
        weight_scale,
        alpha_and_bias,
        output,
    )


__all__ = ["run_sm120_nvfp4_conv3d"]
