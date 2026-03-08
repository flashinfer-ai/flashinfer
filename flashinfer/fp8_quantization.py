import functools
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit.fp8_quantization import gen_mxfp8_quantization_sm100_module
from .utils import (
    device_support_pdl,
    register_custom_op,
    register_fake_op,
)


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
        is_sf_swizzled_layout: bool = True,
        alignment: int = 32,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor to MxFP8 format.

        Args:
            input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
            is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.
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
            if is_sf_swizzled_layout:
                out_sf_size = _compute_swizzled_layout_sf_size(
                    input.shape[0], input.shape[1] // 32, 128
                )
            else:
                out_sf_size = input.numel() // 32
            out_sf = torch.empty((out_sf_size,), dtype=torch.uint8, device=input.device)
            module.mxfp8_quantize_host(
                input,
                out_val,
                out_sf,
                is_sf_swizzled_layout,
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
            if is_sf_swizzled_layout:
                out_sf_size = _compute_swizzled_layout_sf_size(m, padded_k // 32, 128)
            else:
                out_sf_size = m * padded_k // 32
            out_sf = torch.empty((out_sf_size,), dtype=torch.uint8, device=input.device)
            module.mxfp8_quantize(
                input,
                out_val,
                out_sf,
                is_sf_swizzled_layout,
                alignment,
                enable_pdl,
            )
            return out_val, out_sf

    @register_fake_op("flashinfer::mxfp8_quantize_sm100")
    def _fake_mxfp8_quantize_sm100(
        input: torch.Tensor,
        is_sf_swizzled_layout: bool = True,
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
        is_sf_swizzled_layout: bool = True,
    ) -> torch.Tensor:
        """Dequantize input tensor from MxFP8 format.

        Args:
            input (torch.Tensor): Input tensor of shape [M, K] with dtype FLOAT8_E4M3.
            scale_tensor (torch.Tensor): Scale factors tensor with shape determined by layout and sf_vec_size.
            is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.

        Returns:
            torch.Tensor: Dequantized float tensor of shape [M, K] with dtype float32.
        """
        out = torch.empty(input.shape, dtype=torch.float32, device=input.device)
        module.mxfp8_dequantize_host(
            input,
            scale_tensor,
            out,
            is_sf_swizzled_layout,
        )
        return out

    @register_fake_op("flashinfer::mxfp8_dequantize_host_sm100")
    def _fake_mxfp8_dequantize_host_sm100(
        input: torch.Tensor,
        scale_tensor: torch.Tensor,
        is_sf_swizzled_layout: bool = True,
    ) -> torch.Tensor:
        return input.new_empty([input.shape[0], input.shape[1]], dtype=torch.float32)

    # Register the module
    return SimpleNamespace(
        mxfp8_quantize_sm100=mxfp8_quantize_sm100,
        mxfp8_dequantize_host_sm100=mxfp8_dequantize_host_sm100,
    )


@flashinfer_api
def mxfp8_quantize(
    input: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input tensor to MxFP8 format.

    This function implements MxFP8 quantization that converts input tensors to a compressed MxFP8 format
    with associated scale factors. It supports various input data types and scale factor layouts.

    Args:
        input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
        is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.
        alignment (int, optional): sfVecSize. Defaults to 32.
        enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability. Defaults to None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K] with dtype FLOAT8_E4M3
            - Scale factors tensor with shape determined by layout and sf_vec_size
    """
    sf_vec_size = 32

    assert input.shape[-1] % sf_vec_size == 0
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    x_q, sf = get_mxfp8_quantization_sm100_module().mxfp8_quantize_sm100(
        input,
        is_sf_swizzled_layout,
        alignment,
        enable_pdl,
    )
    return x_q, sf


@flashinfer_api
def mxfp8_dequantize_host(
    input: torch.Tensor,
    scale_tensor: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
) -> torch.Tensor:
    """Dequantize input tensor from MxFP8 format.

    This function performs dequantization by converting a packed FP8 tensor in MxFP8 format
    back to float values using the associated scale factors.

    Args:
        input (torch.Tensor): Packed FP8 tensor in MxFP8 format of shape [M, K] with dtype FLOAT8_E4M3.
        scale_tensor (torch.Tensor): Scale factors tensor with shape determined by layout and sf_vec_size.
        is_sf_swizzled_layout (bool, optional): Whether scale factors use swizzled layout. Defaults to True.

    Returns:
        torch.Tensor: Dequantized float tensor of shape [M, K] with dtype float32.

    """

    return get_mxfp8_quantization_sm100_module().mxfp8_dequantize_host_sm100(
        input,
        scale_tensor,
        is_sf_swizzled_layout,
    )
