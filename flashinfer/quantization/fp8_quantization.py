import functools
from types import SimpleNamespace
from typing import Literal, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.quantize import mxfp8_quantize_trace
from ..jit.fp8_quantization import gen_mxfp8_quantization_sm100_module
from ..utils import (
    device_support_pdl,
    register_custom_op,
    register_fake_op,
)
from ..tllm_enums import SfLayout


def _compute_swizzled_layout_sf_size(total_row, total_column, row_size=128):
    padded_row = (total_row + row_size - 1) // row_size * row_size
    padded_column = (total_column + 3) // 4 * 4
    return padded_row * padded_column


@functools.cache
def get_mxfp8_quantization_sm100_module():
    module = gen_mxfp8_quantization_sm100_module().build_and_load()

    @register_custom_op(
        "flashinfer::mxfp8_quantize_sm100",
        mutates_args=(""),
    )
    def mxfp8_quantize_sm100(
        input: torch.Tensor,
        sf_swizzle_layout: SfLayout = SfLayout.layout_linear,
        alignment: int = 32,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor to MxFP8 format.

        Args:
            input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
            sf_swizzle_layout (SfLayout, optional): Swizzle layout for scale factors. Defaults to SfLayout.layout_linear.
            alignment (int, optional): sfVecSize. Defaults to 32. Note that alignment is not used in the host kernel.
            enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
                If None, automatically detects based on device capability. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Quantized tensor of shape [M, K] with dtype FLOAT8_E4M3
                - Scale factors tensor with shape determined by layout and sf_vec_size
        """
        if input.device.type == "cpu":
            out_val = torch.empty(input.shape, dtype=torch.uint8, device=input.device)
            if sf_swizzle_layout == SfLayout.layout_128x4:
                out_sf_size = _compute_swizzled_layout_sf_size(
                    input.shape[0], input.shape[1] // 32, 128
                )
            elif sf_swizzle_layout == SfLayout.layout_linear:
                out_sf_size = input.numel() // 32
            elif sf_swizzle_layout == SfLayout.layout_8x4:
                raise ValueError(
                    f"{sf_swizzle_layout} is not supported for mxfp8 quantization on CPU."
                )
            else:
                raise ValueError(
                    f"Invalid sf_swizzle_layout value: {sf_swizzle_layout}"
                )
            out_sf = torch.empty((out_sf_size,), dtype=torch.uint8, device=input.device)
            module.mxfp8_quantize_host(
                input,
                out_val,
                out_sf,
                sf_swizzle_layout.value,
            )
            return out_val, out_sf
        else:
            if enable_pdl is None:
                enable_pdl = device_support_pdl(input.device)
            m = input.numel() // input.shape[-1]
            k = input.shape[-1]
            padded_k = (k + alignment - 1) // alignment * alignment
            out_val = torch.empty(
                (*input.shape[:-1], padded_k),
                dtype=torch.float8_e4m3fn,
                device=input.device,
            )
            if sf_swizzle_layout == SfLayout.layout_128x4:
                out_sf_size = _compute_swizzled_layout_sf_size(m, padded_k // 32, 128)
            elif sf_swizzle_layout == SfLayout.layout_8x4:
                out_sf_size = _compute_swizzled_layout_sf_size(m, padded_k // 32, 8)
            elif sf_swizzle_layout == SfLayout.layout_linear:
                out_sf_size = m * padded_k // 32
            else:
                raise ValueError(
                    f"Invalid sf_swizzle_layout value: {sf_swizzle_layout}"
                )
            out_sf = torch.empty((out_sf_size,), dtype=torch.uint8, device=input.device)
            module.mxfp8_quantize(
                input,
                out_val,
                out_sf,
                sf_swizzle_layout.value,
                alignment,
                enable_pdl,
            )
            return out_val, out_sf

    @register_fake_op("flashinfer::mxfp8_quantize_sm100")
    def _fake_mxfp8_quantize_sm100(
        input: torch.Tensor,
        sf_swizzle_layout: SfLayout = SfLayout.layout_linear,
        alignment: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, k = input.shape
        return (
            input.new_empty([m, k], dtype=torch.int64),  # FLOAT8_E4M3
            input.new_empty([m * k // 32], dtype=torch.int32),  # Scale factors
        )

    @register_custom_op(
        "flashinfer::mxfp8_dequantize_host_sm100",
        mutates_args=("",),
    )
    def mxfp8_dequantize_host_sm100(
        input: torch.Tensor,
        scale_tensor: torch.Tensor,
        sf_swizzle_layout: SfLayout = SfLayout.layout_linear,
    ) -> torch.Tensor:
        """Dequantize input tensor from MxFP8 format.

        Args:
            input (torch.Tensor): Input tensor of shape [M, K] with dtype FLOAT8_E4M3.
            scale_tensor (torch.Tensor): Scale factors tensor with shape determined by layout and sf_vec_size.
            sf_swizzle_layout (SfLayout, optional): Swizzle layout for scale factors. Defaults to SfLayout.layout_linear.

        Returns:
            torch.Tensor: Dequantized float tensor of shape [M, K] with dtype float32.
        """
        out = torch.empty(input.shape, dtype=torch.float32, device=input.device)
        module.mxfp8_dequantize_host(
            input,
            scale_tensor,
            out,
            sf_swizzle_layout.value,
        )
        return out

    @register_fake_op("flashinfer::mxfp8_dequantize_host_sm100")
    def _fake_mxfp8_dequantize_host_sm100(
        input: torch.Tensor,
        scale_tensor: torch.Tensor,
        sf_swizzle_layout: SfLayout = SfLayout.layout_linear,
    ) -> torch.Tensor:
        return input.new_empty([input.shape[0], input.shape[1]], dtype=torch.float32)

    # Register the module
    return SimpleNamespace(
        mxfp8_quantize_sm100=mxfp8_quantize_sm100,
        mxfp8_dequantize_host_sm100=mxfp8_dequantize_host_sm100,
    )


@flashinfer_api(trace=mxfp8_quantize_trace)
def mxfp8_quantize(
    input: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    enable_pdl: Optional[bool] = None,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
    sf_swizzle_layout: Optional[SfLayout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input tensor to MxFP8 format.

    This function implements MxFP8 quantization that converts input tensors to a compressed MxFP8 format
    with associated scale factors. It supports various input data types and scale factor layouts.

    Args:
        input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
        is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.
        alignment (int, optional): sfVecSize. Defaults to 32.
        enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability (SM >= 9.0). Defaults to None.
        backend (Literal["cuda", "cute-dsl"], optional): Backend to use for quantization. Options are:
            - "cuda": Use JIT-compiled CUDA kernel (default, stable)
            - "cute-dsl": Use CuTe-DSL kernel (requires SM100+, **experimental**)
        sf_swizzle_layout (Optional[SfLayout], optional): Swizzle layout for scale factors.
            If provided,it overrides is_sf_swizzled_layout. Defaults to None.
            The SfLayout.layout_8x4 is only available for 'cuda' backend.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K] with dtype FLOAT8_E4M3
            - Scale factors tensor with shape determined by layout and sf_vec_size

    Warning:
        The "cute-dsl" backend is **experimental** and not part of the stable API.
        It may change or be removed in future versions without notice.
        Use at your own risk for production workloads.
    """
    sf_vec_size = 32

    assert input.shape[-1] % sf_vec_size == 0
    assert backend in ("cuda", "cute-dsl"), (
        f"backend must be 'cuda' or 'cute-dsl', got '{backend}'"
    )

    if sf_swizzle_layout is None:
        sf_swizzle_layout = (
            SfLayout.layout_128x4 if is_sf_swizzled_layout else SfLayout.layout_linear
        )

    if backend == "cute-dsl":
        from ..cute_dsl import is_cute_dsl_available

        if not is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend requested but CuTe-DSL is not available. "
                "Please install nvidia-cutlass-dsl package."
            )
        from .kernels.mxfp8_quantize import mxfp8_quantize_cute_dsl

        if sf_swizzle_layout == SfLayout.layout_8x4:
            raise ValueError("SfLayout.layout_8x4 is not supported in cute-dsl backend")

        return mxfp8_quantize_cute_dsl(
            input,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment,
            enable_pdl=enable_pdl,
        )
    else:
        # backend == "cuda"
        if enable_pdl is None:
            enable_pdl = device_support_pdl(input.device)
        x_q, sf = get_mxfp8_quantization_sm100_module().mxfp8_quantize_sm100(
            input,
            sf_swizzle_layout,
            alignment,
            enable_pdl,
        )
        return x_q, sf


@flashinfer_api
def mxfp8_dequantize_host(
    input: torch.Tensor,
    scale_tensor: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    sf_swizzle_layout: Optional[SfLayout] = None,
) -> torch.Tensor:
    """Dequantize input tensor from MxFP8 format.

    This function performs dequantization by converting a packed FP8 tensor in MxFP8 format
    back to float values using the associated scale factors.

    Args:
        input (torch.Tensor): Packed FP8 tensor in MxFP8 format of shape [M, K] with dtype FLOAT8_E4M3.
        scale_tensor (torch.Tensor): Scale factors tensor with shape determined by layout and sf_vec_size.
        is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.
        sf_swizzle_layout (Optional[SfLayout], optional): Swizzle layout for scale factors.
            If provided,it overrides is_sf_swizzled_layout. Defaults to None.
            Available options are 1. SfLayout.layout_128x4; 2. SfLayout.layout_linear.

    Returns:
        torch.Tensor: Dequantized float tensor of shape [M, K] with dtype float32.

    """

    if sf_swizzle_layout is None:
        sf_swizzle_layout = (
            SfLayout.layout_128x4 if is_sf_swizzled_layout else SfLayout.layout_linear
        )
    return get_mxfp8_quantization_sm100_module().mxfp8_dequantize_host_sm100(
        input,
        scale_tensor,
        sf_swizzle_layout,
    )
