import functools
import math
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


def _mxfp8_sf_shape_2d(m, k, sf_swizzle_layout, alignment=32):
    """Rank-preserving 2D shape of the scale-factor buffer for one ``[m, k]`` matrix.

    Mirrors the (row, column) padding the kernel applies to the flat 1D buffer, so
    reshaping the 1D output to this shape is a pure view.  For the swizzled layouts
    the row dim is padded to the swizzle block size and the column dim to a multiple
    of 4 (matching ``_compute_swizzled_layout_sf_size``); ``layout_linear`` is
    unpadded.  This is the 2D scale convention ``nvfp4_quantize`` already returns.
    """
    padded_k = (k + alignment - 1) // alignment * alignment
    cols = padded_k // 32
    if sf_swizzle_layout == SfLayout.layout_128x4:
        return ((m + 127) // 128 * 128, (cols + 3) // 4 * 4)
    if sf_swizzle_layout == SfLayout.layout_8x4:
        return ((m + 7) // 8 * 8, (cols + 3) // 4 * 4)
    return (m, cols)  # layout_linear


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
            out_sf = torch.zeros((out_sf_size,), dtype=torch.uint8, device=input.device)
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
    rank_preserving: bool = False,
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
        rank_preserving (bool, optional): Scale-factor output shape convention. Defaults to False.
            - ``False`` (default, legacy): the scale factor is returned as a flat 1D
              buffer regardless of input rank; for batched ``[B, M, K]`` input the batch
              is folded into M (``B*M``) and the swizzle padding is applied to the
              combined ``B*M`` dimension.
            - ``True`` (opt-in, recommended): the scale factor's rank mirrors the input.
              A 2D ``[M, K]`` input returns a 2D ``[M_pad, K_blocks_pad]`` scale (the same
              convention :func:`nvfp4_quantize` uses), and a 3D ``[B, M, K]`` input returns
              a 3D ``[B, M_pad, K_blocks_pad]`` scale that is padded **per batch** (each
              batch's M padded to the swizzle block size independently). The per-batch form
              is what batched consumers such as the cuDNN ``bmm`` path require. This is the
              transition path toward unifying the mxfp8 (1D) and nvfp4 (2D) scale
              conventions; the default will flip to ``True`` in a future release.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K] with dtype FLOAT8_E4M3
            - Scale factors tensor with shape determined by layout, sf_vec_size and
              ``rank_preserving`` (see above)

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

    if rank_preserving and backend != "cuda":
        raise NotImplementedError(
            "rank_preserving=True is currently only supported with backend='cuda'."
        )

    # Batched (>=3D) swizzled input with rank_preserving must be quantized per batch:
    # the default kernel folds the batch into M and pads the combined B*M, so the
    # per-batch padding rows simply don't exist in that flat buffer and cannot be
    # recovered by a reshape. (2D input and the linear layout need no per-batch padding
    # and are handled by the reshape at the end.)
    if (
        rank_preserving
        and input.dim() >= 3
        and sf_swizzle_layout != SfLayout.layout_linear
    ):
        if enable_pdl is None:
            enable_pdl = device_support_pdl(input.device)
        *batch_dims, m, k = input.shape
        bsz = math.prod(batch_dims)
        flat_in = input.reshape(bsz, m, k)
        module = get_mxfp8_quantization_sm100_module()
        qs, sfs = [], []
        for i in range(bsz):
            q_i, sf_i = module.mxfp8_quantize_sm100(
                flat_in[i].contiguous(), sf_swizzle_layout, alignment, enable_pdl
            )
            qs.append(q_i)
            sfs.append(sf_i)
        padded_k = (k + alignment - 1) // alignment * alignment
        m_pad, cols = _mxfp8_sf_shape_2d(m, k, sf_swizzle_layout, alignment)
        x_q = torch.stack(qs, 0).reshape(*batch_dims, m, padded_k)
        sf = torch.stack(sfs, 0).reshape(*batch_dims, m_pad, cols)
        return x_q, sf

    if backend == "cute-dsl":
        from ..cute_dsl import is_cute_dsl_available

        if not is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend requested but CuTe-DSL is not available. "
                "Please install nvidia-cutlass-dsl package."
            )
        from .kernels.mxfp8_quantize import mxfp8_quantize_cute_dsl

        is_sf_swizzled_layout_cute = sf_swizzle_layout != SfLayout.layout_linear
        is_sf_8x4_layout_cute = sf_swizzle_layout == SfLayout.layout_8x4

        return mxfp8_quantize_cute_dsl(
            input,
            is_sf_swizzled_layout=is_sf_swizzled_layout_cute,
            alignment=alignment,
            enable_pdl=enable_pdl,
            is_sf_8x4_layout=is_sf_8x4_layout_cute,
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
        if rank_preserving:
            # 2D input -> 2D [M_pad, K_blocks_pad]; 3D+ here is necessarily the linear
            # layout (swizzled 3D returned above), which needs no per-batch padding so a
            # plain reshape mirroring the input rank is exact.
            *batch_dims, m, k = input.shape
            m_pad, cols = _mxfp8_sf_shape_2d(m, k, sf_swizzle_layout, alignment)
            sf = (
                sf.reshape(*batch_dims, m, cols)
                if batch_dims
                else sf.reshape(m_pad, cols)
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
