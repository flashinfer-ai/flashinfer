import functools
from types import SimpleNamespace
from typing import Literal, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..cutile import is_cuda_tile_available
from ..trace.templates.quantize import (
    mxfp8_grouped_quantize_trace,
    mxfp8_quantize_trace,
)
from ..jit.fp8_quantization import gen_mxfp8_quantization_sm100_module
from ..utils import (
    device_support_pdl,
    get_compute_capability,
    register_custom_op,
    register_fake_op,
)
from ..tllm_enums import SfLayout


def _round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize input tensor to MxFP8 format.

    Implements MxFP8 quantization that converts input tensors to a
    compressed MxFP8 format with associated scale factors.  Supports
    various input data types and scale-factor layouts.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``[M, K]`` with dtype fp16/bf16/fp8_quantized.
    is_sf_swizzled_layout : bool
        Whether to use the swizzled layout for scale factors.  Defaults to
        ``True``.
    alignment : int
        ``sfVecSize``.  Defaults to ``32``.
    enable_pdl : bool, optional
        Whether to enable Programmatic Dependent Launch.  Auto-detected
        from device capability (SM >= 9.0) when ``None``.
    backend : {"cuda", "cute-dsl"}
        Backend to use:

        - ``"cuda"``: stable JIT-compiled CUDA kernel (default).
        - ``"cute-dsl"``: CuTe-DSL kernel (SM100+, **experimental**).
    sf_swizzle_layout : SfLayout, optional
        Swizzle layout for scale factors; when supplied this overrides
        ``is_sf_swizzled_layout``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(x_q, sf)`` where ``x_q`` has shape ``[M, K]`` with dtype
        ``FLOAT8_E4M3`` and ``sf`` is the scale-factor tensor whose
        shape depends on the chosen layout and ``sf_vec_size`` (fixed at
        ``32`` here).

    Warnings
    --------
    The ``"cute-dsl"`` backend is **experimental** and not part of the
    stable API.  It may change or be removed in future versions without
    notice.
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
        return x_q, sf


@functools.cache
def get_mxfp8_grouped_quantization_module():
    """Register the grouped MXFP8 quantization op and its fake (meta) twin.

    Returns a namespace with two implementations:

    - ``mxfp8_grouped_quantize_impl``: the real op, registered via
      ``register_custom_op`` and backed by the cuTile kernel. It quantizes a
      ``[B, M, K]`` batch to MXFP8 with UE8M0 block scales and lays out the
      outputs for the FlashInfer masked grouped GEMM.
    - ``_fake_mxfp8_grouped_quantize``: the metadata-only twin registered via
      ``register_fake_op`` for tracing and shape inference; it allocates
      outputs with the correct shapes, dtypes, and strides without launching
      any kernel.

    The factory is cached so registration happens once. The cuTile kernel is
    arch-independent and compiles lazily on its first launch, so there is no
    module to build and load here.
    """

    @register_custom_op(
        "flashinfer::mxfp8_grouped_quantize",
        mutates_args=("",),
    )
    def mxfp8_grouped_quantize_impl(
        a: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the grouped MXFP8 cuTile quantizer and lay out the outputs for
        the FlashInfer masked grouped GEMM. Assumes inputs are already
        validated by the public ``mxfp8_grouped_quantize`` wrapper.
        """
        from .kernels.cutile.mxfp8_grouped_quantize_cutile import (
            mxfp8_grouped_quantize_cutile,
        )

        b, m, k = a.shape
        padded_k = _round_up(k, 128)
        padded_m = _round_up(m, 128)
        scale_k = padded_k // 32

        if padded_k == k:
            input_tensor = a.contiguous()
        else:
            input_tensor = a.new_zeros((b, m, padded_k))
            input_tensor[:, :, :k] = a

        output = torch.empty(
            (b, m, padded_k),
            dtype=torch.float8_e4m3fn,
            device=a.device,
        )
        output_scales = torch.empty(
            (b, padded_m, scale_k),
            dtype=torch.uint8,
            device=a.device,
        )

        problem_sizes = torch.empty((b, 3), dtype=torch.int32, device=a.device)
        problem_sizes[:, 0] = mask
        problem_sizes[:, 1] = 0
        problem_sizes[:, 2] = padded_k
        group_ids = torch.arange(b, dtype=torch.int32, device=a.device)
        expert_offsets = group_ids * m
        blockscale_offsets = group_ids * padded_m

        mxfp8_grouped_quantize_cutile(
            input_tensor.view(b * m, padded_k),
            problem_sizes,
            expert_offsets,
            blockscale_offsets,
            output.view(b * m, padded_k),
            output_scales,
        )

        output = output.permute(1, 2, 0)
        output_scales = output_scales.view(b, padded_m // 128, scale_k // 4, 32, 4, 4)
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return output, output_scales

    @register_fake_op("flashinfer::mxfp8_grouped_quantize")
    def _fake_mxfp8_grouped_quantize(
        a: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Metadata only: allocate empty outputs with the same shapes, dtypes,
        # and (post-permute) strides as the real op without launching cuTile.
        b, m, k = a.shape
        padded_k = _round_up(k, 128)
        padded_m = _round_up(m, 128)
        scale_k = padded_k // 32

        output = torch.empty(
            (b, m, padded_k),
            dtype=torch.float8_e4m3fn,
            device=a.device,
        )
        output_scales = torch.empty(
            (b, padded_m, scale_k),
            dtype=torch.uint8,
            device=a.device,
        )

        output = output.permute(1, 2, 0)
        output_scales = output_scales.view(b, padded_m // 128, scale_k // 4, 32, 4, 4)
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return output, output_scales

    return SimpleNamespace(
        mxfp8_grouped_quantize_impl=mxfp8_grouped_quantize_impl,
        _fake_mxfp8_grouped_quantize=_fake_mxfp8_grouped_quantize,
    )


@flashinfer_api(trace=mxfp8_grouped_quantize_trace)
def mxfp8_grouped_quantize(
    a: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize grouped inputs to MXFP8 with UE8M0 block scales.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor of shape ``[B, M, K]`` with dtype ``float16`` or
        ``bfloat16``.
    mask : torch.Tensor
        Int32 CUDA tensor of shape ``[B]``. Each value gives the number of
        valid rows to quantize for the corresponding group, and must satisfy
        ``0 <= mask[i] <= M``. This precondition is the caller's
        responsibility: it is not validated at runtime, because reading the
        device-side ``mask`` values would force a host synchronization and break
        CUDA-graph capture. Out-of-range values are undefined behavior. The
        kernel writes scale factors with bounds checking disabled, so
        ``mask[i] > M`` corrupts neighboring groups or writes out of bounds.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(x_q, sf)`` where ``x_q`` has logical shape
        ``[M, padded_K, B]`` with dtype ``float8_e4m3fn`` and ``sf`` has
        logical shape ``[32, 4, padded_M // 128, 4, padded_K // 128, B]``
        with dtype ``uint8``. ``padded_K`` rounds ``K`` up to a multiple
        of 128. The physical layouts are grouped by ``B`` and then
        permuted to match FlashInfer masked grouped GEMM conventions.

        Only the first ``mask[i]`` rows of group ``i`` are written;
        rows ``>= mask[i]`` (and their scale factors) are unspecified.
        The consumer must use the same ``mask`` and read only the valid rows.
    """

    if a.dim() != 3:
        raise ValueError("a must be a 3D tensor with shape [B, M, K]")
    if not a.is_cuda:
        raise ValueError("a must be a CUDA tensor")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a dtype must be torch.float16 or torch.bfloat16")
    if mask.dim() != 1 or mask.size(0) != a.size(0):
        raise ValueError("mask must be a 1D tensor with one entry per group")
    if mask.dtype != torch.int32:
        raise ValueError("mask dtype must be torch.int32")
    if not mask.is_cuda:
        raise ValueError("mask must be a CUDA tensor")
    if mask.device != a.device:
        raise ValueError("mask must live on the same CUDA device as a")

    major, _ = get_compute_capability(a.device)
    if major < 10:
        raise RuntimeError("mxfp8_grouped_quantize requires SM100 or newer")

    if not is_cuda_tile_available():
        raise RuntimeError(
            "mxfp8_grouped_quantize requires the cuTile backend; install "
            "cuda-tile>=1.4.0 and ensure the tileiras compiler is available."
        )

    _, _, k = a.shape
    if k % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {k}")

    return get_mxfp8_grouped_quantization_module().mxfp8_grouped_quantize_impl(a, mask)


@flashinfer_api
def mxfp8_dequantize_host(
    input: torch.Tensor,
    scale_tensor: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    sf_swizzle_layout: Optional[SfLayout] = None,
) -> torch.Tensor:
    r"""Host-side dequantization of an MxFP8 tensor back to float32.

    Performs dequantization by converting a packed FP8 tensor in MxFP8
    format back to float values using the associated scale factors.

    Parameters
    ----------
    input : torch.Tensor
        Packed FP8 tensor in MxFP8 format of shape ``[M, K]`` with dtype
        ``FLOAT8_E4M3``.
    scale_tensor : torch.Tensor
        Scale-factor tensor (shape depends on layout and ``sf_vec_size``).
    is_sf_swizzled_layout : bool
        Whether the scale factors are stored in the swizzled layout.
        Defaults to ``True``.
    sf_swizzle_layout : SfLayout, optional
        Explicit swizzle layout for scale factors; when supplied this
        overrides ``is_sf_swizzled_layout``.  Options are
        :attr:`SfLayout.layout_128x4` and :attr:`SfLayout.layout_linear`.

    Returns
    -------
    torch.Tensor
        Dequantized float tensor of shape ``[M, K]`` with dtype
        ``float32``.
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
