import functools
from functools import cache
from types import SimpleNamespace
from typing import Any, Tuple

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm100a_nvcc_flags
from .utils import register_custom_op, register_fake_op

_fp4_quantization_sm100 = None


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
    global _fp4_quantization_sm100
    if _fp4_quantization_sm100 is None:
        module = gen_fp4_quantization_sm100_module().build_and_load()

        @register_custom_op(
            "flashinfer::fp4_quantize_sm100",
            mutates_args=(""),
        )
        def fp4_quantize_sm100(
            input: torch.Tensor,
            global_scale: torch.Tensor,
            sf_vec_size: int = 16,
            sf_use_ue8m0: bool = False,
            is_sf_swizzled_layout: bool = True,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Quantize input tensor to FP4 format.

            Args:
                input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
                global_scale (torch.Tensor): Global scale factor of shape [1] and dtype float32.
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
            global_scale: torch.Tensor,
            sf_vec_size: int = 16,
            sf_use_ue8m0: bool = False,
            is_sf_swizzled_layout: bool = True,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            m, k = input.shape
            return (
                input.new_empty([m, k // 2], dtype=torch.int64),  # FLOAT4_E2M1X2
                input.new_empty(
                    [m * k // sf_vec_size], dtype=torch.int32
                ),  # Scale factors
            )

        # Register the module
        _fp4_quantization_sm100 = SimpleNamespace(
            fp4_quantize_sm100=fp4_quantize_sm100,
        )

    return _fp4_quantization_sm100


def fp4_quantize(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input tensor to FP4 format.

    This function implements FP4 quantization that converts input tensors to a compressed FP4 format
    with associated scale factors. It supports various input data types and scale factor layouts.

    Args:
        input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
        global_scale (torch.Tensor): Global scale factor of shape [1] and dtype float32.
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
    if sf_vec_size != 16:
        raise NotImplementedError("sf_vec_size can only be 16")

    return get_fp4_quantization_sm100_module().fp4_quantize_sm100(
        input,
        global_scale,
        sf_vec_size,
        sf_use_ue8m0,
        is_sf_swizzled_layout,
    )
