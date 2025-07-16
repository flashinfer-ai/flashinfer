"""
Copyright (c) 2025 by FlashInfer team.

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

import functools
from functools import cache
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm100a_nvcc_flags
from .utils import register_custom_op, register_fake_op


def _pad_scale_factors(
    unswizzled_sf: torch.Tensor, m: int, n: int, sf_vec_size: int = 16
) -> torch.Tensor:
    """Pad scale factors tensor to meet alignment requirements.

    Args:
        unswizzled_sf (torch.Tensor): Input scale factors tensor with dtype uint8.
        m (int): M dimension.
        n (int): N dimension.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.

    Returns:
        torch.Tensor: Padded scale factors tensor.
    """
    factor = sf_vec_size * 4
    padded_row = ((m + 128 - 1) // 128) * 128  # Next multiple of 128
    padded_col = ((n + factor - 1) // factor) * factor  # Next multiple of 64

    # Pad the input tensor to [padded_row, padded_col // scaling_vector_size]
    pad_rows = padded_row - m
    pad_cols = (padded_col - n) // sf_vec_size
    if pad_rows == 0 and pad_cols == 0:
        return unswizzled_sf
    else:
        return torch.nn.functional.pad(
            unswizzled_sf, (0, pad_cols, 0, pad_rows), mode="constant", value=0
        ).contiguous()


def gen_fp4_quantization_sm100_module() -> JitSpec:
    return gen_jit_spec(
        "fp4_quantization_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/thop/fp4Quantize.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/kernels/quantization.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP8",
        ],
        extra_cflags=[
            "-DENABLE_BF16",
            "-DENABLE_FP8",
        ],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
        ],
    )


@functools.cache
def get_fp4_quantization_sm100_module():
    module = gen_fp4_quantization_sm100_module().build_and_load()

    @register_custom_op(
        "flashinfer::fp4_quantize_sm100",
        mutates_args=(""),
    )
    def fp4_quantize_sm100(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor to FP4 format.

        Args:
            input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
            global_scale (torch.Tensor, optional): Global scale factor of shape [1] and dtype float32.
            sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
            sf_use_ue8m0 (bool, optional): Whether to use UE8M0 format for scale factors. Defaults to False.
            is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Quantized tensor of shape [M, K/2] with dtype FLOAT4_E2M1X2
                - Scale factors tensor with shape determined by layout and sf_vec_size
        """
        return module.fp4_quantize(
            input,
            global_scale,
            sf_vec_size,
            sf_use_ue8m0,
            is_sf_swizzled_layout,
        )

    @register_fake_op("flashinfer::fp4_quantize_sm100")
    def _fake_fp4_quantize_sm100(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, k = input.shape
        return (
            input.new_empty([m, k // 2], dtype=torch.int64),  # FLOAT4_E2M1X2
            input.new_empty([m * k // sf_vec_size], dtype=torch.int32),  # Scale factors
        )

    @register_custom_op(
        "flashinfer::block_scale_interleave_sm100",
        mutates_args=("",),
    )
    def block_scale_interleave_sm100(
        unswizzled_sf: torch.Tensor,
    ) -> torch.Tensor:
        """Swizzle block scale tensor for FP4 format.

        Args:
            unswizzled_sf (torch.Tensor): unswizzled block scale tensor with dtype uint8.

        Returns:
            torch.Tensor: output tensor for swizzled block scale with dtype uint8.
        """
        return module.block_scale_interleave(
            unswizzled_sf,
        )

    @register_fake_op("flashinfer::block_scale_interleave_sm100")
    def _fake_block_scale_interleave_sm100(
        unswizzled_sf: torch.Tensor,
    ) -> torch.Tensor:
        return unswizzled_sf.new_empty(
            [unswizzled_sf.shape[0] * unswizzled_sf.shape[1] // 16], dtype=torch.uint8
        )

    @register_custom_op(
        "flashinfer::e2m1_and_ufp8sf_scale_to_float_sm100",
        mutates_args=(""),
    )
    def e2m1_and_ufp8sf_scale_to_float_sm100(
        e2m1_tensor: torch.Tensor,
        ufp8_scale_tensor: torch.Tensor,
        global_scale_tensor: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        ufp8_type: int = 1,
        is_sf_swizzled_layout: bool = True,
    ) -> torch.Tensor:
        """Convert E2M1 format tensor and UFP8 scale factors to float tensor.

        This function performs dequantization by converting a packed FP4 tensor in E2M1 format
        back to float values using the associated UFP8 scale factors and global scale.

        Args:
            e2m1_tensor (torch.Tensor): Packed FP4 tensor in E2M1 format of shape [M, K/2] with dtype uint8.
            ufp8_scale_tensor (torch.Tensor): Scale factors tensor in UFP8 format with dtype uint8.
            global_scale_tensor (torch.Tensor, optional): Global scale factor of shape [1] and dtype float32.
            sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
            ufp8_type (int, optional): UFP8 scale factor type (0 for UE8M0, 1 for E4M3). Defaults to 1.
            is_sf_swizzled_layout (bool, optional): Whether scale factors use swizzled layout. Defaults to True.

        Returns:
            torch.Tensor: Dequantized float tensor of shape [M, K] with dtype float32.
        """
        return module.e2m1_and_ufp8sf_scale_to_float(
            e2m1_tensor.cpu(),
            ufp8_scale_tensor.cpu().reshape(-1),
            global_scale_tensor.cpu(),
            sf_vec_size,
            ufp8_type,
            is_sf_swizzled_layout,
        )

    @register_fake_op("flashinfer::e2m1_and_ufp8sf_scale_to_float_sm100")
    def _fake_e2m1_and_ufp8sf_scale_to_float_sm100(
        e2m1_tensor: torch.Tensor,
        ufp8_scale_tensor: torch.Tensor,
        global_scale_tensor: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        ufp8_type: int = 1,
        is_sf_swizzled_layout: bool = True,
    ) -> torch.Tensor:
        return e2m1_tensor.new_empty(
            [e2m1_tensor.shape[0], e2m1_tensor.shape[1] * 2], dtype=torch.float32
        )

    # Register the module
    return SimpleNamespace(
        fp4_quantize_sm100=fp4_quantize_sm100,
        block_scale_interleave_sm100=block_scale_interleave_sm100,
        e2m1_and_ufp8sf_scale_to_float_sm100=e2m1_and_ufp8sf_scale_to_float_sm100,
    )


def fp4_quantize(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input tensor to FP4 format.

    This function implements FP4 quantization that converts input tensors to a compressed FP4 format
    with associated scale factors. It supports various input data types and scale factor layouts.

    Args:
        input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
        global_scale (torch.Tensor, optional): Global scale factor of shape [1] and dtype float32.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
        sf_use_ue8m0 (bool, optional): Whether to use UE8M0 format for scale factors. Defaults to False.
        is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size

    Raises:
        NotImplementedError: If any of the following features are requested but not implemented:
            - BFloat16 input when BFloat16 is not enabled
            - FP8 input when FP8 is not enabled
            - sf_vec_size other than 16
    """
    if sf_vec_size != 16 and sf_vec_size != 32:
        raise NotImplementedError("sf_vec_size can only be 16 or 32")

    assert input.shape[-1] % sf_vec_size == 0
    x_q, sf = get_fp4_quantization_sm100_module().fp4_quantize_sm100(
        input,
        global_scale,
        sf_vec_size,
        sf_use_ue8m0,
        is_sf_swizzled_layout,
    )
    sf = sf.reshape((-1, input.shape[-1] // sf_vec_size))
    return x_q, sf


def block_scale_interleave(unswizzled_sf: torch.Tensor) -> torch.Tensor:
    """Swizzle block scale tensor for FP4 format.

    This function swizzles the block scale tensor to optimize memory access patterns
    for FP4 operations. The output needs to be padded in the m dimension to be a multiple of 128.

    Args:
        unswizzled_sf (torch.Tensor): Input tensor with dtype uint8.

    Returns:
        torch.Tensor: Swizzled tensor with the same shape as input.

    Raises:
        AssertionError: If input dtype is not uint8.
    """
    # TODO(shuw): check input dtype is uint8
    assert (
        unswizzled_sf.dtype == torch.uint8
    ), f"Input dtype must be uint8, got {unswizzled_sf.dtype}"
    return get_fp4_quantization_sm100_module().block_scale_interleave_sm100(
        unswizzled_sf,
    )


def e2m1_and_ufp8sf_scale_to_float(
    e2m1_tensor: torch.Tensor,
    ufp8_scale_tensor: torch.Tensor,
    global_scale_tensor: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    ufp8_type: int = 1,
    is_sf_swizzled_layout: bool = True,
) -> torch.Tensor:
    """Convert E2M1 format tensor and UFP8 scale factors to float tensor.

    This function performs dequantization by converting a packed FP4 tensor in E2M1 format
    back to float values using the associated UFP8 scale factors and global scale.

    Args:
        e2m1_tensor (torch.Tensor): Packed FP4 tensor in E2M1 format of shape [M, K/2] with dtype uint8.
        ufp8_scale_tensor (torch.Tensor): Scale factors tensor in UFP8 format with dtype uint8.
        global_scale_tensor (torch.Tensor, optional): Global scale factor of shape [1] and dtype float32.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
        ufp8_type (int, optional): UFP8 scale factor type (0 for UE8M0, 1 for E4M3). Defaults to 1.
        is_sf_swizzled_layout (bool, optional): Whether scale factors use swizzled layout. Defaults to True.

    Returns:
        torch.Tensor: Dequantized float tensor of shape [M, K] with dtype float32.

    """

    return get_fp4_quantization_sm100_module().e2m1_and_ufp8sf_scale_to_float_sm100(
        e2m1_tensor,
        ufp8_scale_tensor,
        global_scale_tensor,
        sf_vec_size,
        ufp8_type,
        is_sf_swizzled_layout,
    )
