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
from enum import Enum
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit import JitSpec
from .jit import env as jit_env
from .jit import (
    gen_jit_spec,
    sm121a_nvcc_flags,
    sm120a_nvcc_flags,
    sm110a_nvcc_flags,
    sm103a_nvcc_flags,
    sm100a_nvcc_flags,
    sm90a_nvcc_flags,
)
from .jit.cpp_ext import is_cuda_version_at_least
from .utils import (
    device_support_pdl,
    get_shuffle_matrix_a_row_indices,
    get_shuffle_matrix_sf_a_row_indices,
    register_custom_op,
    register_fake_op,
    get_compute_capability,
)


def _compute_swizzled_layout_sf_size(total_row, total_column, row_size=128):
    padded_row = (total_row + row_size - 1) // row_size * row_size
    padded_column = (total_column + 3) // 4 * 4
    return padded_row * padded_column


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
    return gen_fp4_quantization_module(sm100a_nvcc_flags, "100")


def gen_fp4_quantization_sm103_module() -> JitSpec:
    return gen_fp4_quantization_module(sm103a_nvcc_flags, "103")


def gen_fp4_quantization_sm90_module() -> JitSpec:
    return gen_fp4_quantization_module(sm90a_nvcc_flags, "90")


def gen_fp4_quantization_sm110_module() -> JitSpec:
    return gen_fp4_quantization_module(sm110a_nvcc_flags, "110")


def gen_fp4_quantization_sm120_module() -> JitSpec:
    return gen_fp4_quantization_module(sm120a_nvcc_flags, "120")


def gen_fp4_quantization_sm121_module() -> JitSpec:
    return gen_fp4_quantization_module(sm121a_nvcc_flags, "121")


def gen_fp4_quantization_module(nvcc_flags: List[str], device_arch: str) -> JitSpec:
    return gen_jit_spec(
        f"fp4_quantization_{device_arch}",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/thop/fp4Quantize.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/tensorrt_llm/thop/fp4Op.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/kernels/quantization.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
        ],
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4" if is_cuda_version_at_least("12.8") else "",
        ],
        extra_cflags=[
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4" if is_cuda_version_at_least("12.8") else "",
        ],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
        ],
    )


@functools.cache
def get_fp4_quantization_module(backend: str = "100"):
    backend_modules = {
        "121": gen_fp4_quantization_sm121_module,
        "120": gen_fp4_quantization_sm120_module,
        "110": gen_fp4_quantization_sm110_module,
        "103": gen_fp4_quantization_sm103_module,
        "100": gen_fp4_quantization_sm100_module,
        "90": gen_fp4_quantization_sm90_module,
    }

    if backend not in backend_modules:
        raise ValueError(f"Invalid backend: {backend}")

    module = backend_modules[backend]().build_and_load()

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
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor to FP4 format.

        Args:
            input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
            global_scale (torch.Tensor, optional): Global scale factor of shape [1] and dtype float32.
            sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
            sf_use_ue8m0 (bool, optional): Whether to use UE8M0 format for scale factors. Defaults to False.
            is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.
            is_sf_8x4_layout (bool, optional): Whether to use 8x4 layout or 128x4 layout for scale factors. Defaults to False.
            enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
                If None, automatically detects based on device capability. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Quantized tensor of shape [M, K/2] with dtype FLOAT4_E2M1X2
                - Scale factors tensor with shape determined by layout and sf_vec_size
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(input.device)
        out_val = torch.empty(
            (*input.shape[:-1], input.shape[-1] // 2),
            dtype=torch.uint8,
            device=input.device,
        )
        m = input.numel() // input.shape[-1]
        k = input.shape[-1]
        if is_sf_swizzled_layout:
            out_sf_size = _compute_swizzled_layout_sf_size(
                m, k // sf_vec_size, 8 if is_sf_8x4_layout else 128
            )
        else:
            out_sf_size = m * k // sf_vec_size
        out_sf = torch.empty((out_sf_size,), dtype=torch.uint8, device=input.device)
        module.fp4_quantize(
            input,
            global_scale,
            out_val,
            out_sf,
            sf_vec_size,
            sf_use_ue8m0,
            is_sf_swizzled_layout,
            is_sf_8x4_layout,
            enable_pdl,
        )
        return out_val, out_sf

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
        "flashinfer::mxfp4_dequantize_host",
        mutates_args=(""),
    )
    def mxfp4_dequantize_host(
        weight: torch.Tensor,
        scale: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        out = torch.empty(
            (weight.shape[0], weight.shape[1] * 2),
            dtype=torch.float32,
            device=weight.device,
        )
        module.mxfp4_dequantize_host(
            weight,
            scale,
            out,
            group_size,
        )
        return out

    @register_fake_op("flashinfer::mxfp4_dequantize_host")
    def _fake_mxfp4_dequantize_host(
        weight: torch.Tensor,
        scale: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        return weight.new_empty(
            [weight.shape[0], weight.shape[1] * 2], dtype=torch.float32
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
            unswizzled_sf (torch.Tensor): unswizzled block scale tensor with dtype uint8 or bfloat16.

        Returns:
            torch.Tensor: output tensor for swizzled block scale with dtype uint8 or bfloat16.
        """
        num_experts = unswizzled_sf.shape[0] if unswizzled_sf.dim() == 3 else 1
        expert_out_size = _compute_swizzled_layout_sf_size(
            unswizzled_sf.shape[-2], unswizzled_sf.shape[-1], 128
        )
        out = torch.empty(
            (num_experts * expert_out_size,),
            dtype=unswizzled_sf.dtype,
            device=unswizzled_sf.device,
        )
        module.block_scale_interleave_sm100(unswizzled_sf, out)
        return out

    @register_fake_op("flashinfer::block_scale_interleave_sm100")
    def _fake_block_scale_interleave_sm100(
        unswizzled_sf: torch.Tensor,
    ) -> torch.Tensor:
        return unswizzled_sf.new_empty(
            [unswizzled_sf.shape[0] * unswizzled_sf.shape[1] // 16], dtype=torch.uint8
        )

    @register_custom_op(
        "flashinfer::fp4_batched_quantize_sm100",
        mutates_args=("",),
    )
    def fp4_batched_quantize_sm100(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a batched tensor to FP4 (E2M1x2) with per-block scale factors.

        This function converts a float/bfloat16 (or FP8-quantized) input tensor into a
        packed FP4 tensor using the E2M1 format (two 4-bit values per byte), along with
        per-block scale factors. Scale factors are encoded as UE4M3 by default, or UE8M0
        when requested, and an optional global scale can be applied.

        Args:
            input (torch.Tensor): Input tensor of shape [B, M, K] with dtype torch.float16,
                torch.bfloat16, or an FP8-quantized dtype supported by the kernel.
            global_scale (torch.Tensor, optional): Global scale factor of shape [1] and
                dtype float32.
            sf_vec_size (int, optional): Scale-factor vector size and alignment unit along K.
                Supported/expected values:
                - 16 (NVFP4 path; supported)
                - 32 (MXFP4 path; not supported yet)
                Defaults to 16.
            sf_use_ue8m0 (bool, optional): Scale-factor encoding type.
                False → UE4M3 (default), True → UE8M0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - self_fp4 (torch.Tensor): Packed FP4 tensor in E2M1x2 format of shape
                [B, M, K // 2] with dtype torch.uint8 (two FP4 lanes per byte).
                - self_block_scale_factors (torch.Tensor): Block scale factors with dtype
                uint8 (UE4M3 or UE8M0), laid out as a flat buffer of shape
                [B, ceil(M / 128) * 128 * ceil(K / sf_vec_size / 4) * 4].

        Notes:
            - K must be even (because outputs pack two FP4 values per byte).
            - For best performance, K should be a multiple of sf_vec_size; the scale-factor
            buffer is aligned to sf_vec_size along K, pads M to multiples of 128, and
            rounds (K / sf_vec_size) up to a multiple of 4 for storage.
            - The batch dimension B is preserved for both outputs.
        """
        b, m, k = input.shape
        out_val = torch.empty(
            (b, m, k // 2),
            dtype=torch.uint8,
            device=input.device,
        )
        out_sf = torch.empty(
            (b, _compute_swizzled_layout_sf_size(m, k // sf_vec_size, 128)),
            dtype=torch.uint8,
            device=input.device,
        )
        module.fp4_batched_quantize(
            input,
            global_scale,
            out_val,
            out_sf,
            sf_vec_size,
            sf_use_ue8m0,
        )
        return out_val, out_sf

    @register_fake_op("flashinfer::fp4_batched_quantize_sm100")
    def _fake_fp4_batched_quantize_sm100(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, m, k = input.shape
        return (
            input.new_empty([b, m, k // 2], dtype=torch.uint8),  # FLOAT4_E2M1X2
            input.new_empty(
                [b, _compute_swizzled_layout_sf_size(m, k // sf_vec_size, 128)],
                dtype=torch.uint8,
            ),  # swizzled SF buffer
        )

    @register_custom_op(
        "flashinfer::silu_and_mul_scaled_nvfp4_experts_quantize_sm100",
        mutates_args=("",),
    )
    def silu_and_mul_scaled_nvfp4_experts_quantize_sm100(
        input: torch.Tensor,
        mask: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a silu and matmul with masked batched tensor to FP4 (E2M1x2) with per-block scale factors.

        This function first does silu and matmul to a float/bfloat16 input tensor then convect the result
        into a packed FP4 tensor using the E2M1 format (two 4-bit values per byte), along with
        per-block scale factors. Scale factors are encoded as UE4M3 by default, or UE8M0
        when requested, and an optional global scale can be applied.

        Args:
            input (torch.Tensor): Input tensor of shape [B, M, K] with dtype torch.float16,
                torch.bfloat16, or an FP8-quantized dtype supported by the kernel.
            mask (torch.Tensor): mask tensor of shape [B] with dtype torch.int32.
            global_scale (torch.Tensor, optional): Global scale factor of shape [1] and
                dtype float32.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - self_fp4 (torch.Tensor): Packed FP4 tensor in E2M1x2 format of shape
                [B, M, K // 2] with dtype torch.uint8 (two FP4 lanes per byte).
                - self_block_scale_factors (torch.Tensor): Block scale factors with dtype
                uint8 (UE4M3 or UE8M0), laid out as a flat buffer of shape
                [B, ceil(M / 128) * 128 * ceil(K / sf_vec_size / 4) * 4].

        Notes:
            - K must be even (because outputs pack two FP4 values per byte).
            - For best performance, K should be a multiple of sf_vec_size; the scale-factor
            buffer is aligned to sf_vec_size along K, pads M to multiples of 128, and
            rounds (K / sf_vec_size) up to a multiple of 4 for storage.
            - The batch dimension B is preserved for both outputs.
        """
        device = input.device
        l, m, k_by_2 = input.shape
        k = k_by_2 // 2
        sf_vec_size = 16
        assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

        scale_k = k // sf_vec_size
        padded_k = (scale_k + (4 - 1)) // 4 * 4
        padded_k_int32 = padded_k // 4
        padded_m = (m + (128 - 1)) // 128 * 128
        output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
        output_scales = torch.empty(
            l, padded_m, padded_k_int32, device=device, dtype=torch.int32
        )

        module.silu_and_mul_scaled_nvfp4_experts_quantize(
            output.view(l * m, k // 2),
            output_scales.view(l * padded_m, padded_k_int32),
            input.view(l * m, k_by_2),
            global_scale,
            mask,
            True,
        )
        output = output.permute(1, 2, 0)
        output_scales = output_scales.view(torch.float8_e4m3fn).view(
            l, padded_m // 128, padded_k // 4, 32, 4, 4
        )
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return output, output_scales

    @register_fake_op("flashinfer::silu_and_mul_scaled_nvfp4_experts_quantize_sm100")
    def _fake_silu_and_mul_scaled_nvfp4_experts_quantize_sm100(
        input: torch.Tensor,
        mask: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        l, m, k_by_2 = input.shape
        k = k_by_2 // 2
        sf_vec_size = 16
        assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

        scale_k = k // sf_vec_size
        padded_k = (scale_k + (4 - 1)) // 4 * 4
        padded_k_int32 = padded_k // 4
        padded_m = (m + (128 - 1)) // 128 * 128
        output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
        output_scales = torch.empty(
            l, padded_m, padded_k_int32, device=device, dtype=torch.int32
        )

        output_scales = output_scales.view(torch.float8_e4m3fn).view(
            l, padded_m // 128, padded_k // 4, 32, 4, 4
        )
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return (output, output_scales)

    @register_custom_op(
        "flashinfer::scaled_fp4_grouped_quant_sm100",
        mutates_args=("",),
    )
    def scaled_fp4_grouped_quant_sm100(
        input_tensor: torch.Tensor,
        input_global_scale: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor to FP4 and return quantized tensor and scale, for
        grouped gemm inputs (e.g., grouped_gemm_nt_masked for flashinfer).
        Args:
            input: The input tensor to be quantized to FP4, with shape (l, m, k)
                l is number of groups, m is number of tokens per group, k is number of features.
            input_global_scale: A scalar scaling factor for the entire tensor, with
                shape (l,).
        Outputs:
            output: The quantized tensor in FP4, with shape (m, k // 2, l) but the physical
                layout is (l, m, k // 2). `// 2` is because two fp4 values are packed into
                an uint8.
            output_scales: The blockscale tensor in FP8-E4M3, with shape (32, 4, rm, 4, rk, l)
                but the physical layout is (l, rm, rk, 32, 4, 4).
        Note:
            For the shape of output_scales, `32 * 4 * rm` is a padded m to nearest multiple of 128.
            `4 * rk` is a padded `k // 16` to nearest multiple of 4. These layout constants are
            required by the NVIDIA Blackwell MMA operations.
        """
        device = input_tensor.device
        l, m, k = input_tensor.shape
        sf_vec_size = 16
        assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

        scale_k = k // sf_vec_size
        padded_k = (scale_k + (4 - 1)) // 4 * 4
        padded_k_int32 = padded_k // 4
        padded_m = (m + (128 - 1)) // 128 * 128
        output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
        output_scales = torch.empty(
            l, padded_m, padded_k_int32, device=device, dtype=torch.int32
        )

        module.silu_and_mul_scaled_nvfp4_experts_quantize(
            output.view(l * m, k // 2),
            output_scales.view(l * padded_m, padded_k_int32),
            input_tensor.view(l * m, k),
            input_global_scale,
            mask,
            False,
        )
        # The physical layout of the output is (l, m, k // 2), but we want to return a
        # logical layout (m, k // 2, l) required by the flashinfer masked group gemm.
        output = output.permute(1, 2, 0)
        # The physical layout of the output scales is already swizzled as (l, rm, rk, 32, 4, 4), a
        # requirement for the flashinfer masked group gemm, where rm=m/128 and rk=k/4. The logic
        # layout is (32, 4, rm, 4, rk, l).
        output_scales = output_scales.view(torch.float8_e4m3fn).view(
            l, padded_m // 128, padded_k // 4, 32, 4, 4
        )
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return output, output_scales

    @register_fake_op("flashinfer::scaled_fp4_grouped_quant_sm100")
    def _fake_scaled_fp4_grouped_quant_sm100(
        input_tensor: torch.Tensor,
        input_global_scale: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_tensor.device
        l, m, k = input_tensor.shape
        sf_vec_size = 16
        assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

        scale_k = k // sf_vec_size
        padded_k = (scale_k + (4 - 1)) // 4 * 4
        padded_k_int32 = padded_k // 4
        padded_m = (m + (128 - 1)) // 128 * 128
        output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
        output_scales = torch.empty(
            l, padded_m, padded_k_int32, device=device, dtype=torch.int32
        )

        output = output.permute(1, 2, 0)
        output_scales = output_scales.view(torch.float8_e4m3fn).view(
            l, padded_m // 128, padded_k // 4, 32, 4, 4
        )
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return output, output_scales

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
        out = torch.zeros(
            (e2m1_tensor.shape[0], e2m1_tensor.shape[1] * 2),
            dtype=torch.float32,
            device="cpu",
        )
        module.e2m1_and_ufp8sf_scale_to_float_sm100(
            e2m1_tensor.cpu(),
            ufp8_scale_tensor.cpu().reshape(-1),
            global_scale_tensor.cpu(),
            out,
            sf_vec_size,
            ufp8_type,
            is_sf_swizzled_layout,
        )
        return out

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
        mxfp4_dequantize_host=mxfp4_dequantize_host,
        fp4_batched_quantize_sm100=fp4_batched_quantize_sm100,
        silu_and_mul_scaled_nvfp4_experts_quantize_sm100=silu_and_mul_scaled_nvfp4_experts_quantize_sm100,
        scaled_fp4_grouped_quant_sm100=scaled_fp4_grouped_quant_sm100,
    )


@flashinfer_api
def fp4_quantize(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
    enable_pdl: Optional[bool] = None,
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
        is_sf_8x4_layout (bool, optional): Whether to use 8x4 layout or 128x4 layout for scale factors. Defaults to False.
        enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size

    Raises:
        NotImplementedError: If any of the following features are requested but not implemented:
            - BFloat16 input when BFloat16 is not enabled
            - FP8 input when FP8 is not enabled
            - sf_vec_size other than 16 or 32
    """
    if sf_vec_size != 16 and sf_vec_size != 32:
        raise NotImplementedError("sf_vec_size can only be 16 or 32")

    # for column major input, we need to transpose the input
    is_column_major = input.stride(-2) == 1
    if is_column_major:
        input = input.transpose(-2, -1)

    assert input.shape[-1] % sf_vec_size == 0
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    # get input device sm version
    major, minor = get_compute_capability(input.device)
    x_q, sf = get_fp4_quantization_module(f"{major}{minor}").fp4_quantize_sm100(
        input,
        global_scale,
        sf_vec_size,
        sf_use_ue8m0,
        is_sf_swizzled_layout,
        is_sf_8x4_layout,
        enable_pdl,
    )
    sf = sf.reshape((-1, input.shape[-1] // sf_vec_size))
    if is_column_major:
        x_q = x_q.transpose(-2, -1)
        sf = sf.transpose(-2, -1)

    return x_q, sf


@flashinfer_api
def block_scale_interleave(unswizzled_sf: torch.Tensor) -> torch.Tensor:
    """Swizzle block scale tensor for FP4 format.

    This function swizzles the block scale tensor to optimize memory access patterns
    for FP4 operations. The output needs to be padded in the m dimension to be a multiple of 128.

    Args:
        unswizzled_sf (torch.Tensor): Input tensor with dtype uint8 or bfloat16.

    Returns:
        torch.Tensor: Swizzled tensor with the same shape as input.

    Raises:
        AssertionError: If input dtype is not uint8 or bfloat16.
    """
    # TODO(shuw): check input dtype is uint8
    assert (
        unswizzled_sf.dtype == torch.uint8 or unswizzled_sf.dtype == torch.bfloat16
    ), f"Input dtype must be uint8 or bfloat16, got {unswizzled_sf.dtype}"

    major, minor = get_compute_capability(unswizzled_sf.device)
    device_arch = f"{major * 10 + minor}"

    return get_fp4_quantization_module(device_arch).block_scale_interleave_sm100(
        unswizzled_sf,
    )


# Maintain compatibility with libraries using the old name
nvfp4_block_scale_interleave = block_scale_interleave


@flashinfer_api
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
    # NOTE(Zihao): this is another cpu op, should decouple it from cuda ops in the future
    major, minor = get_compute_capability(
        torch.device("cuda:0")
    )  # select any cuda device to get a compute capability
    device_arch = f"{major * 10 + minor}"
    return get_fp4_quantization_module(
        device_arch
    ).e2m1_and_ufp8sf_scale_to_float_sm100(
        e2m1_tensor,
        ufp8_scale_tensor,
        global_scale_tensor,
        sf_vec_size,
        ufp8_type,
        is_sf_swizzled_layout,
    )


@flashinfer_api
def shuffle_matrix_a(input_tensor: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    """
    PyTorch equivalent of trtllm-gen `shuffleMatrixA`
    """
    row_indices = get_shuffle_matrix_a_row_indices(input_tensor, epilogue_tile_m)

    return input_tensor[row_indices.to(input_tensor.device)]


@flashinfer_api
def shuffle_matrix_sf_a(
    input_tensor: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: int = 16,
):
    """
    Cuda implementation of trtllm-gen `shuffleMatrixSfA` but with a caveat.
    `shuffleMatrixSfA` expects the input to be in 128x4 layout and then
    apply the same shuffling in `shuffleMatrixA` and writes out in 128x4
    layout.
    This function expects the input to be in linear layout. It's done this
    way because the scaling factors in the NVFP4 checkpoints are quantized
    and are in linear layout.
    This function doesn't add padding.
    """

    row_indices = get_shuffle_matrix_sf_a_row_indices(input_tensor, epilogue_tile_m)

    w_shuffled = input_tensor[row_indices.to(input_tensor.device)]

    # 128x4
    return block_scale_interleave(w_shuffled)


class SfLayout(Enum):
    """
    Layout of scale factors for NVFP4.
    """

    layout_128x4 = 0
    layout_8x4 = 1
    layout_linear = 2


@flashinfer_api
def nvfp4_quantize(
    a,
    a_global_sf,
    sfLayout=SfLayout.layout_128x4,
    do_shuffle=False,
    sf_vec_size=16,
    enable_pdl=None,
):
    """
    Quantize input tensor to NVFP4 format.

    Parameters:
        a (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16.
        a_global_sf (torch.Tensor): Global scale factor of shape [1] with dtype float32.
        sfLayout (SfLayout, optional): Scale factor layout. Defaults to SfLayout.layout_128x4.
        do_shuffle (bool, optional): Whether to shuffle the scale factors. Defaults to False. Only TRTLLM backend needs to shuffle the tensor B scale factors.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
        enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size
    """

    if do_shuffle:
        # Weights 128x4 + shuffle. It is done during the model load and we do not care much about the perf
        assert sfLayout == SfLayout.layout_128x4
        a_fp4, a_sf = fp4_quantize(
            a.cuda(),
            a_global_sf.cuda(),
            sf_vec_size,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
            is_sf_8x4_layout=False,
            enable_pdl=enable_pdl,
        )

        epilogue_tile_m = 128
        a_fp4 = shuffle_matrix_a(a_fp4.view(torch.uint8), epilogue_tile_m)
        a_sf = shuffle_matrix_sf_a(a_sf.view(torch.uint8), epilogue_tile_m).reshape(
            a_sf.shape
        )
    else:
        # Activations with 8x4 layout for SFs (GEMM with small tileN)
        # Activations with 128x4 layout for SFs (GEMM with large tileN)
        a_fp4, a_sf = fp4_quantize(
            a.cuda(),
            a_global_sf.cuda(),
            sf_vec_size,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=True,
            is_sf_8x4_layout=sfLayout == SfLayout.layout_8x4,
            enable_pdl=enable_pdl,
        )

    return a_fp4, a_sf


@flashinfer_api
def mxfp4_quantize(a):
    """
    Quantize input tensor to MXFP4 format.

    Parameters:
        a (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K/2] with dtype uint8 (FLOAT4_E2M1X2)
            - Scale factors tensor with shape determined by layout and sf_vec_size (uint8)
    """
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    a_fp4, a_sf = fp4_quantize(a.cuda(), a_global_sf.cuda(), 32, True, True)
    return a_fp4, a_sf


@flashinfer_api
def mxfp4_dequantize(a_fp4, a_sf):
    """
    Dequantize input tensor from MXFP4 format.

    Parameters:
        a_fp4 (torch.Tensor): Quantized tensor of shape [M, K/2] with dtype uint8 (FLOAT4_E2M1X2)
        a_sf (torch.Tensor): Scale factors tensor with shape determined by layout and sf_vec_size (uint8)

    Returns:
        torch.Tensor: Dequantized tensor of shape [M, K] with dtype float.
    """
    return e2m1_and_ufp8sf_scale_to_float(
        a_fp4.cpu().view(torch.uint8),
        a_sf.cpu().view(torch.uint8).reshape(-1),
        torch.tensor([1.0], device=a_fp4.device),
        32,
        0,
        True,
    )


@flashinfer_api
def mxfp4_dequantize_host(
    weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Dequantize input tensor from MXFP4 format on host.

    Parameters:
        weight (torch.Tensor): Quantized tensor of shape [M, K/2] with dtype uint8 (FLOAT4_E2M1X2)
        scale (torch.Tensor): Scale factors tensor with shape determined by layout and sf_vec_size (uint8)
        group_size (int, optional): Group size for dequantization. Defaults to 32.

    Returns:
        torch.Tensor: Dequantized tensor of shape [M, K] with dtype float.
    """
    # NOTE(Zihao): the cpu op should be decouplied from cuda ops because it's device independent, should refactor this in the future
    major, minor = get_compute_capability(
        torch.device("cuda:0")
    )  # use any cuda device to get a compute capability
    device_arch = f"{major * 10 + minor}"
    return get_fp4_quantization_module(device_arch).mxfp4_dequantize_host(
        weight,
        scale,
        group_size,
    )


@flashinfer_api
def nvfp4_batched_quantize(
    a,
    a_global_sf,
    sf_vec_size=16,
):
    """
    Quantize batched input tensor to NVFP4 format.

    Parameters:
        a (torch.Tensor): Input tensor of shape [B, M, K] with dtype fp16/bf16.
        a_global_sf (torch.Tensor): Global scale factor of shape [1] with dtype float32.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [B, M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size
    """
    major, minor = get_compute_capability(a.device)
    device_arch = f"{major * 10 + minor}"
    a_fp4, a_sf = get_fp4_quantization_module(device_arch).fp4_batched_quantize_sm100(
        a,
        a_global_sf,
        sf_vec_size,
        False,
    )
    return a_fp4, a_sf


@flashinfer_api
def scaled_fp4_grouped_quantize(
    a,
    mask,
    a_global_sf,
):
    """
    quantize batched input tensor to NVFP4 format with mask.
    Parameters:
        a (torch.Tensor): Input tensor of shape [B, M, K] with dtype fp16/bf16.
        a_global_sf (torch.Tensor): Global scale factor of shape [1] with dtype float32.
        mask (torch.Tensor): Mask tensor to apply before quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [B, M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size
    """
    major, minor = get_compute_capability(a.device)
    device_arch = f"{major * 10 + minor}"
    a_fp4, a_sf = get_fp4_quantization_module(
        device_arch
    ).scaled_fp4_grouped_quant_sm100(
        a,
        a_global_sf,
        mask,
    )
    return a_fp4, a_sf
