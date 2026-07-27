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


@lru_cache(maxsize=None)
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
    weight_scale_bytes = (
        ((output_channels + 127) // 128) * (((channels * 27 // 16) + 3) // 4) * 512
    )

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
    while swizzle_size > m_tiles:
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
        device_index=packed_input.device.index,
    )
    compiled(
        packed_input,
        packed_weight,
        input_scale,
        weight_scale.reshape(-1),
        alpha_and_bias,
        output,
    )


__all__ = ["run_sm120_nvfp4_conv3d"]
