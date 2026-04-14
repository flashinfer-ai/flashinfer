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
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import (
    gen_jit_spec,
    sm121a_nvcc_flags,
    sm120a_nvcc_flags,
    sm120f_nvcc_flags,
    sm110a_nvcc_flags,
    sm103a_nvcc_flags,
    sm100a_nvcc_flags,
    sm90a_nvcc_flags,
)
from ..jit.cpp_ext import is_cuda_version_at_least
from ..utils import (
    backend_requirement,
    device_support_pdl,
    get_compute_capability,
    get_shuffle_matrix_a_row_indices,
    get_shuffle_matrix_sf_a_row_indices,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
    round_up,
)
from ..tllm_enums import SfLayout


def _compute_swizzled_layout_sf_size(total_row, total_column, row_size=128):
    padded_row = round_up(total_row, row_size)
    padded_column = round_up(total_column, 4)
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
    padded_row = round_up(m, 128)
    padded_col = round_up(n, factor)

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


def gen_fp4_quantization_sm120f_module() -> JitSpec:
    return gen_fp4_quantization_module(sm120f_nvcc_flags, "120f")


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
        "120f": gen_fp4_quantization_sm120f_module,
        "120": gen_fp4_quantization_sm120_module,
        "110": gen_fp4_quantization_sm110_module,
        "103": gen_fp4_quantization_sm103_module,
        "100": gen_fp4_quantization_sm100_module,
        "90": gen_fp4_quantization_sm90_module,
    }

    # Prefer 'f' (family / feature-set) variant for SM12x when CUDA >= 12.9,
    # as it enables native FP4 conversion instructions (cvt.rn.satfinite.e2m1x2.f32).
    # sm_120f covers the entire SM12x family (both SM120 and SM121).
    # See: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
    if backend in ("120", "121"):
        from ..utils import version_at_least

        if version_at_least(torch.version.cuda, "12.9"):
            backend = "120f"

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
            out_sf_size_padded = out_sf_size
        else:
            out_sf_size = m * k // sf_vec_size
            out_sf_size_padded = round_up(m, 16) * k // sf_vec_size
        out_sf = torch.empty(
            (out_sf_size_padded,), dtype=torch.uint8, device=input.device
        )
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
        return out_val, out_sf[:out_sf_size]

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
        padded_k = round_up(scale_k, 4)
        padded_k_int32 = padded_k // 4
        padded_m = round_up(m, 128)
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
        padded_k = round_up(scale_k, 4)
        padded_k_int32 = padded_k // 4
        padded_m = round_up(m, 128)
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
        padded_k = round_up(scale_k, 4)
        padded_k_int32 = padded_k // 4
        padded_m = round_up(m, 128)
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
        padded_k = round_up(scale_k, 4)
        padded_k_int32 = padded_k // 4
        padded_m = round_up(m, 128)
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
    backend: str = "cuda",
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
        backend (str, optional): Backend to use for quantization.
            - "cuda": Use CUDA kernel (default, stable).
            - "cute-dsl": Use CuTe-DSL kernel (requires SM100+, **experimental**).
              Supported combinations:
              * sf_vec_size=16, sf_use_ue8m0=False: all layouts, fp16/bf16/fp8 (NVFP4)
              * sf_vec_size=32, sf_use_ue8m0=True: 128x4 swizzled and linear, fp16/bf16 (MXFP4)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size

    Raises:
        NotImplementedError: If any of the following features are requested but not implemented:
            - BFloat16 input when BFloat16 is not enabled
            - FP8 input when FP8 is not enabled
            - sf_vec_size other than 16 or 32
        ValueError: If the "cute-dsl" backend is requested for an unsupported parameter combination.

    Warning:
        The "cute-dsl" backend is **experimental** and not part of the stable API.
        It may change or be removed in future versions without notice.
    """
    if sf_vec_size != 16 and sf_vec_size != 32:
        raise NotImplementedError("sf_vec_size can only be 16 or 32")

    if backend == "cute-dsl":
        return _fp4_quantize_cute_dsl(
            input,
            global_scale,
            sf_vec_size,
            sf_use_ue8m0,
            is_sf_swizzled_layout,
            is_sf_8x4_layout,
            enable_pdl,
        )
    elif backend != "cuda":
        raise ValueError(f"Unknown backend: {backend}. Must be 'cuda' or 'cute-dsl'.")

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
    # Swizzled sf includes row/column padding from block_scale_interleave
    # (rows to multiple of 128, cols to multiple of 4), so we use the padded
    # column count and let -1 absorb the padded row count.
    # Non-swizzled sf has exactly m * (k // sf_vec_size) elements, no padding.
    if is_sf_swizzled_layout:
        sf_cols = round_up(input.shape[-1] // sf_vec_size, 4)
        sf = sf.reshape((-1, sf_cols))
    else:
        sf = sf.reshape((-1, input.shape[-1] // sf_vec_size))
    if is_column_major:
        x_q = x_q.transpose(-2, -1)
        sf = sf.transpose(-2, -1)

    return x_q, sf


def _fp4_quantize_cute_dsl(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    sf_vec_size: int,
    sf_use_ue8m0: bool,
    is_sf_swizzled_layout: bool,
    is_sf_8x4_layout: bool,
    enable_pdl: Optional[bool],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CuTe-DSL dispatch for fp4_quantize. Maps parameters to the appropriate kernel."""
    from ..cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        raise RuntimeError(
            "CuTe-DSL backend requested but CuTe-DSL is not available. "
            "Please install the required dependencies."
        )

    if sf_vec_size == 16 and not sf_use_ue8m0:
        # NVFP4 path: E4M3 scale factors, sf_vec_size=16, all layouts
        from .kernels.nvfp4_quantize import (
            SF_LAYOUT_128x4,
            SF_LAYOUT_8x4,
            SF_LAYOUT_LINEAR,
            nvfp4_quantize_cute_dsl,
        )

        if not is_sf_swizzled_layout:
            sf_layout = SF_LAYOUT_LINEAR
        elif is_sf_8x4_layout:
            sf_layout = SF_LAYOUT_8x4
        else:
            sf_layout = SF_LAYOUT_128x4

        return nvfp4_quantize_cute_dsl(
            input, global_scale, sf_layout=sf_layout, enable_pdl=enable_pdl
        )

    elif sf_vec_size == 32 and sf_use_ue8m0:
        # MXFP4 path: UE8M0 scale factors, sf_vec_size=32
        if is_sf_8x4_layout:
            raise ValueError(
                "CuTe-DSL MXFP4 kernel does not support 8x4 layout. "
                "Supported: swizzled 128x4 and linear."
            )
        from .kernels.mxfp4_quantize import (
            SF_LAYOUT_128x4,
            SF_LAYOUT_LINEAR,
            mxfp4_quantize_cute_dsl,
        )

        sf_layout = SF_LAYOUT_128x4 if is_sf_swizzled_layout else SF_LAYOUT_LINEAR
        return mxfp4_quantize_cute_dsl(
            input, sf_layout=sf_layout, enable_pdl=enable_pdl
        )

    else:
        raise ValueError(
            f"CuTe-DSL backend does not support sf_vec_size={sf_vec_size} with "
            f"sf_use_ue8m0={sf_use_ue8m0}. Supported: "
            f"(sf_vec_size=16, sf_use_ue8m0=False) for NVFP4, "
            f"(sf_vec_size=32, sf_use_ue8m0=True) for MXFP4."
        )


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


@flashinfer_api
def nvfp4_quantize(
    a,
    a_global_sf,
    sfLayout=SfLayout.layout_128x4,
    do_shuffle=False,
    sf_vec_size=16,
    enable_pdl=None,
    backend: str = "cuda",
):
    """
    Quantize input tensor to NVFP4 format.

    Parameters:
        a (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/float8_e4m3fn.
        a_global_sf (torch.Tensor): Global scale factor of shape [1] with dtype float32.
        sfLayout (SfLayout, optional): Scale factor layout. Defaults to SfLayout.layout_128x4.
        do_shuffle (bool, optional): Whether to shuffle the scale factors. Defaults to False. Only TRTLLM backend needs to shuffle the tensor B scale factors.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
        enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability. Defaults to None.
        backend (str, optional): Backend to use for quantization.
            - "cuda": Use CUDA kernel (default, stable)
            - "cute-dsl": Use CuTe-DSL kernel (requires SM100+, **experimental**).
              Supports all sfLayout values (layout_128x4, layout_8x4, layout_linear).
              Supports input dtypes: fp16, bf16, float8_e4m3fn.
              Only supports sf_vec_size=16.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size

    Warning:
        The "cute-dsl" backend is **experimental** and not part of the stable API.
        It may change or be removed in future versions without notice.
    """
    if backend == "cuda":
        if do_shuffle:
            assert sfLayout == SfLayout.layout_128x4
            is_sf_swizzled_layout = False
            is_sf_8x4_layout = False
        else:
            is_sf_swizzled_layout = sfLayout != SfLayout.layout_linear
            is_sf_8x4_layout = sfLayout == SfLayout.layout_8x4

        a_fp4, a_sf = fp4_quantize(
            a.cuda(),
            a_global_sf.cuda(),
            sf_vec_size,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            is_sf_8x4_layout=is_sf_8x4_layout,
            enable_pdl=enable_pdl,
        )
    elif backend == "cute-dsl":
        from ..cute_dsl import is_cute_dsl_available

        if not is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend requested but CuTe-DSL is not available. "
                "Please install the required dependencies."
            )
        if sf_vec_size != 16:
            raise ValueError(
                f"CuTe-DSL backend only supports sf_vec_size=16, got {sf_vec_size}"
            )
        from .kernels.nvfp4_quantize import (
            SF_LAYOUT_128x4,
            SF_LAYOUT_8x4,
            SF_LAYOUT_LINEAR,
            nvfp4_quantize_cute_dsl,
        )

        _sf_layout_map = {
            SfLayout.layout_128x4: SF_LAYOUT_128x4,
            SfLayout.layout_8x4: SF_LAYOUT_8x4,
            SfLayout.layout_linear: SF_LAYOUT_LINEAR,
        }
        if do_shuffle:
            assert sfLayout == SfLayout.layout_128x4
            sf_layout_int = SF_LAYOUT_LINEAR
        else:
            sf_layout_int = _sf_layout_map[sfLayout]

        a_fp4, a_sf = nvfp4_quantize_cute_dsl(
            a.cuda(), a_global_sf.cuda(), sf_layout=sf_layout_int, enable_pdl=enable_pdl
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'cuda' or 'cute-dsl'.")

    if do_shuffle:
        epilogue_tile_m = 128
        a_fp4 = shuffle_matrix_a(a_fp4.view(torch.uint8), epilogue_tile_m)
        a_sf = shuffle_matrix_sf_a(a_sf.view(torch.uint8), epilogue_tile_m).reshape(
            a_sf.shape
        )

    return a_fp4, a_sf


@flashinfer_api
def mxfp4_quantize(
    a: torch.Tensor,
    backend: str = "cuda",
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to MXFP4 format.

    Parameters:
        a (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16.
        backend (str, optional): Backend to use for quantization.
            - "cuda": Use CUDA kernel (default, stable)
            - "cute-dsl": Use CuTe-DSL kernel (requires SM100+, **experimental**)
        enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic
            Dependent Launch). Only used when backend="cute-dsl".
            If None, automatically detects based on device capability.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K/2] with dtype uint8 (FLOAT4_E2M1X2)
            - Scale factors tensor with shape determined by layout and sf_vec_size (uint8)

    Warning:
        The "cute-dsl" backend is **experimental** and not part of the stable API.
        It may change or be removed in future versions without notice.
        Use at your own risk for production workloads.
    """
    if backend == "cute-dsl":
        from ..cute_dsl import is_cute_dsl_available

        if not is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend requested but CuTe-DSL is not available. "
                "Please install the required dependencies."
            )
        from .kernels.mxfp4_quantize import mxfp4_quantize_cute_dsl

        return mxfp4_quantize_cute_dsl(a, enable_pdl=enable_pdl)
    elif backend == "cuda":
        a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
        a_fp4, a_sf = fp4_quantize(a.cuda(), a_global_sf.cuda(), 32, True, True)
        return a_fp4, a_sf
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'cuda' or 'cute-dsl'.")


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
def nvfp4_quantize_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_layout: str = "HND",
    k_global_sf: Optional[torch.Tensor] = None,
    v_global_sf: Optional[torch.Tensor] = None,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    float,
    float,
]:
    """Quantize paged KV cache to NVFP4 format for trtllm-gen MHA.

    Quantizes BF16/FP16 K/V caches to NVFP4 with two-level scaling
    (global FP32 + per-block FP8), and swizzles scale factors
    for the SM100 trtllm-gen MHA kernel layout.

    Args:
        k_cache: Key cache tensor.
            HND layout: [num_pages, num_kv_heads, page_size, head_dim].
            NHD layout: [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Value cache tensor (same layout as k_cache).
        kv_layout: Layout of the input KV cache, either ``"HND"`` or ``"NHD"``.
        k_global_sf: Optional global scale factor for K (float32 scalar tensor).
            If None, auto-computed as ``FLOAT8_E4M3_MAX / k_amax``.
        v_global_sf: Optional global scale factor for V (float32 scalar tensor).
            If None, auto-computed as ``FLOAT8_E4M3_MAX / v_amax``.

    Returns:
        kv_cache_fp4: Tuple of (k_fp4, v_fp4) in the same layout as input,
            with head_dim replaced by head_dim//2, dtype=uint8.
        kv_cache_sf: Tuple of (k_scales, v_scales). `k_scales` keeps the linear
            input layout, while `v_scales` uses TRT-LLM's 4-token interleaved
            layout. Both tensors replace `head_dim` with `head_dim//16` and use
            dtype=float8_e4m3fn.
        k_global_scale: Global scale for K (float), equal to ``1 / k_global_sf``.
        v_global_scale: Global scale for V (float), equal to ``1 / v_global_sf``.
    """
    _FLOAT8_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
    # Extract dimensions based on layout
    if kv_layout == "NHD":
        num_pages, page_size, num_kv_heads, head_dim = k_cache.shape
    else:
        num_pages, num_kv_heads, page_size, head_dim = k_cache.shape

    device = k_cache.device
    scale_dim = head_dim // 16

    # Compute global scale factors if not provided.
    # global_sf = FLOAT8_E4M3_MAX / tensor_amax (no *E2M1_MAX factor).
    # This will let the fp4_quantize kernel output the scale factor in range [0, FLOAT8_E4M3_MAX/E2M1_MAX]
    if k_global_sf is None:
        k_amax = k_cache.float().abs().amax()
        k_global_sf = torch.tensor(
            [_FLOAT8_E4M3_MAX / max(k_amax.item(), 1e-12)],
            dtype=torch.float32,
            device=device,
        )
    if v_global_sf is None:
        v_amax = v_cache.float().abs().amax()
        v_global_sf = torch.tensor(
            [_FLOAT8_E4M3_MAX / max(v_amax.item(), 1e-12)],
            dtype=torch.float32,
            device=device,
        )

    # Flatten to 2D [total_tokens, head_dim] for fp4_quantize
    # Both layouts flatten identically since total elements are the same
    k_2d = k_cache.reshape(-1, head_dim)
    v_2d = v_cache.reshape(-1, head_dim)

    # Quantize using FlashInfer's GPU kernel with linear scale layout
    k_packed, k_sf = fp4_quantize(
        k_2d, k_global_sf, sf_vec_size=16, is_sf_swizzled_layout=False
    )
    v_packed, v_sf = fp4_quantize(
        v_2d, v_global_sf, sf_vec_size=16, is_sf_swizzled_layout=False
    )

    # fp4_quantize returns uint8 packed FP4 and uint8 scale factors (FP8 E4M3 encoded)
    # Reshape packed data and scale factors back to the original layout
    if kv_layout == "NHD":
        out_shape_fp4 = (num_pages, page_size, num_kv_heads, head_dim // 2)
        out_shape_sf = (num_pages, page_size, num_kv_heads, scale_dim)
    else:
        out_shape_fp4 = (num_pages, num_kv_heads, page_size, head_dim // 2)
        out_shape_sf = (num_pages, num_kv_heads, page_size, scale_dim)

    kv_cache_fp4 = (
        k_packed.view(torch.uint8).reshape(out_shape_fp4),
        v_packed.view(torch.uint8).reshape(out_shape_fp4),
    )

    # Reshape scale factors (FP8 E4M3 encoded as uint8)
    k_sf_fp8 = k_sf.view(torch.float8_e4m3fn).reshape(out_shape_sf)
    v_sf_fp8 = v_sf.view(torch.float8_e4m3fn).reshape(out_shape_sf)

    # Apply V scale factor swizzling for SM100 trtllm-gen MHA kernel.
    # The swizzle interleaves the token dimension by groups of 4 within each
    # [page_size, head_dim//16] tile per page/head:
    #   output[..., (t//4)*4*S + s*4 + t%4] = input[..., t*S + s]
    # This matches TRT-LLM's quantizeAndWriteFP4KVCache() V swizzle pattern.
    # K scale factors do NOT need swizzling — the kernel reads them with real strides.
    if page_size % 4 != 0 or head_dim % 64 != 0:
        raise ValueError(
            "V-scale swizzling requires page_size % 4 == 0 and head_dim % 64 == 0, "
            f"got page_size={page_size}, head_dim={head_dim}."
        )
    if kv_layout == "NHD":
        swizzle_shape = (
            num_pages,
            page_size // 4,
            4,
            num_kv_heads,
            4,
            scale_dim // 4,
        )
        swizzle_perm = (0, 1, 4, 3, 5, 2)
    else:
        swizzle_shape = (
            num_pages,
            num_kv_heads,
            page_size // 4,
            4,
            4,
            scale_dim // 4,
        )
        swizzle_perm = (0, 1, 2, 4, 5, 3)

    v_sf_fp8 = (
        v_sf_fp8.reshape(swizzle_shape)
        .permute(swizzle_perm)
        .reshape(out_shape_sf)
        .contiguous()
    )

    kv_cache_sf = (k_sf_fp8, v_sf_fp8)

    # Return the inverse of global_sf: global_scale = 1 / global_sf = amax / 448
    k_gs_ret = (1.0 / k_global_sf).item()
    v_gs_ret = (1.0 / v_global_sf).item()

    return kv_cache_fp4, kv_cache_sf, k_gs_ret, v_gs_ret


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


# ---------------------------------------------------------------------------
# NVFP4 KV cache quant/dequant with linear (non-swizzled) block scale layout
# ---------------------------------------------------------------------------


@functools.cache
def get_fp4_kv_dequantization_module():
    from ..jit.fp4_kv_dequantization import gen_fp4_kv_dequantization_module

    module = gen_fp4_kv_dequantization_module().build_and_load()

    @register_custom_op(
        "flashinfer::nvfp4_kv_dequant",
        mutates_args=("output",),
    )
    def nvfp4_kv_dequant(
        fp4_data: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        module.nvfp4_kv_dequant(fp4_data, block_scales, global_scale, output)

    @register_fake_op("flashinfer::nvfp4_kv_dequant")
    def _fake_nvfp4_kv_dequant(
        fp4_data: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        pass

    return SimpleNamespace(nvfp4_kv_dequant=nvfp4_kv_dequant)


@functools.cache
def get_fp4_kv_quantization_module():
    from ..jit.fp4_kv_quantization import gen_fp4_kv_quantization_module

    module = gen_fp4_kv_quantization_module().build_and_load()

    @register_custom_op(
        "flashinfer::nvfp4_kv_quant",
        mutates_args=("fp4_output", "block_scales"),
    )
    def nvfp4_kv_quant(
        input: torch.Tensor,
        global_scale: torch.Tensor,
        fp4_output: torch.Tensor,
        block_scales: torch.Tensor,
    ) -> None:
        module.nvfp4_kv_quant(input, global_scale, fp4_output, block_scales)

    @register_fake_op("flashinfer::nvfp4_kv_quant")
    def _fake_nvfp4_kv_quant(
        input: torch.Tensor,
        global_scale: torch.Tensor,
        fp4_output: torch.Tensor,
        block_scales: torch.Tensor,
    ) -> None:
        pass

    return SimpleNamespace(nvfp4_kv_quant=nvfp4_kv_quant)


_NVFP4_BLOCK_SIZE = 16


@supported_compute_capability([80, 86, 89, 90, 100, 103, 110, 120, 121])
def _nvfp4_kv_dequant_check(fp4_data, block_scales, global_scale, output_dtype=None):
    return True


@backend_requirement({}, common_check=_nvfp4_kv_dequant_check)
@flashinfer_api
def nvfp4_kv_dequantize(
    fp4_data: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GPU dequantization of NVFP4 KV cache data with linear block scale layout.

    Requires SM80+.

    Args:
        fp4_data (torch.Tensor): Packed FP4 data of shape ``[M, K/2]`` with dtype uint8.
        block_scales (torch.Tensor): Per-block FP8 E4M3 scales of shape ``[M, K/16]``
            with dtype uint8.
        global_scale (torch.Tensor): Global scale factor of shape ``[1]`` with dtype float32,
            on the same CUDA device as fp4_data.
        output_dtype (torch.dtype): Output dtype, either ``torch.bfloat16`` or ``torch.float16``.

    Returns:
        torch.Tensor: Dequantized tensor of shape ``[M, K]`` with the specified output dtype.
    """
    M = fp4_data.size(0)
    K = fp4_data.size(1) * 2
    if K % _NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by {_NVFP4_BLOCK_SIZE}")
    output = torch.empty((M, K), dtype=output_dtype, device=fp4_data.device)
    get_fp4_kv_dequantization_module().nvfp4_kv_dequant(
        fp4_data, block_scales, global_scale, output
    )
    return output


@supported_compute_capability([100, 103, 110, 120, 121])
def _nvfp4_kv_quant_check(input, global_scale):
    return True


@backend_requirement({}, common_check=_nvfp4_kv_quant_check)
@flashinfer_api
def nvfp4_kv_quantize(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU quantization to NVFP4 KV cache format with linear block scale layout.

    Requires SM100+ (Blackwell) for the cvt.rn.satfinite.e2m1x2.f32 PTX instruction.

    Args:
        input (torch.Tensor): Input tensor of shape [M, K] with dtype bf16 or fp16.
            K must be divisible by 16.
        global_scale (torch.Tensor): Global scale factor of shape ``[1]`` with dtype float32,
            on the same CUDA device as input.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - fp4_output: Packed FP4 data of shape ``[M, K/2]`` with dtype uint8.
            - block_scales: Per-block FP8 E4M3 scales of shape ``[M, K/16]`` with dtype uint8.
    """
    M, K = input.shape
    if K % _NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by {_NVFP4_BLOCK_SIZE}")
    fp4_output = torch.empty((M, K // 2), dtype=torch.uint8, device=input.device)
    block_scales = torch.empty(
        (M, K // _NVFP4_BLOCK_SIZE), dtype=torch.uint8, device=input.device
    )
    get_fp4_kv_quantization_module().nvfp4_kv_quant(
        input, global_scale, fp4_output, block_scales
    )
    return fp4_output, block_scales
