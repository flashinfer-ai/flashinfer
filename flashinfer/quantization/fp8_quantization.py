import functools
from types import SimpleNamespace
from typing import Dict, Literal, Optional, Tuple

import torch
import tvm_ffi

from ..api_logging import flashinfer_api
from ..cutile import is_cuda_tile_available
from ..trace.templates.quantize import (
    mxfp8_grouped_quantize_trace,
    mxfp8_quantize_trace,
    silu_and_mul_mxfp8_experts_quantize_trace,
)
from ..jit.cake_grouped_mxfp8_quantize import (
    BF16 as _CAKE_GROUPED_MXFP8_BF16,
    BULK_BF16 as _CAKE_GROUPED_MXFP8_BULK_BF16,
    BULK_F16 as _CAKE_GROUPED_MXFP8_BULK_F16,
    F16 as _CAKE_GROUPED_MXFP8_F16,
    TILED_BF16 as _CAKE_GROUPED_MXFP8_TILED_BF16,
    TILED_F16 as _CAKE_GROUPED_MXFP8_TILED_F16,
    TILED_K2048_BF16 as _CAKE_GROUPED_MXFP8_TILED_K2048_BF16,
    TILED_K2048_F16 as _CAKE_GROUPED_MXFP8_TILED_K2048_F16,
    TILED_K4096_BF16 as _CAKE_GROUPED_MXFP8_TILED_K4096_BF16,
    TILED_K4096_F16 as _CAKE_GROUPED_MXFP8_TILED_K4096_F16,
    TILED_WIDE_BF16 as _CAKE_GROUPED_MXFP8_TILED_WIDE_BF16,
    TILED_WIDE_F16 as _CAKE_GROUPED_MXFP8_TILED_WIDE_F16,
)
from ..jit.cake_program import CakeProgram
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


# The masked MXFP8 experts kernel reads its input with 32-byte vector loads.
_MXFP8_EXPERTS_INPUT_ALIGNMENT = 32


def _mxfp8_experts_outputs(
    input: torch.Tensor, use_silu_and_mul: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the physical ``[B, M, padded_K]`` values and
    ``[B, padded_M, padded_K // 32]`` scales of the masked MXFP8 experts kernel."""
    b, m, k = input.shape
    if use_silu_and_mul:
        k //= 2
    padded_k = _round_up(k, 128)
    return (
        input.new_empty((b, m, padded_k), dtype=torch.float8_e4m3fn),
        input.new_empty((b, _round_up(m, 128), padded_k // 32), dtype=torch.uint8),
    )


def _to_masked_gemm_layout(
    x_q: torch.Tensor, sf: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permute the physical kernel outputs to the masked grouped GEMM layout."""
    b, padded_m, scale_k = sf.shape
    sf = sf.view(b, padded_m // 128, scale_k // 4, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)
    return x_q.permute(1, 2, 0), sf


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

    def _mxfp8_experts_quantize(
        input: torch.Tensor,
        mask: torch.Tensor,
        use_silu_and_mul: bool,
        enable_pdl: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_val, out_sf = _mxfp8_experts_outputs(input, use_silu_and_mul)
        # Skip empty inputs; the FFI contiguity check rejects their strides.
        if input.numel() > 0:
            module.silu_and_mul_mxfp8_experts_quantize(
                input, mask, out_val, out_sf, use_silu_and_mul, enable_pdl
            )
        return _to_masked_gemm_layout(out_val, out_sf)

    @register_custom_op(
        "flashinfer::silu_and_mul_mxfp8_experts_quantize_sm100",
        mutates_args=(),
    )
    def silu_and_mul_mxfp8_experts_quantize_sm100(
        input: torch.Tensor,
        mask: torch.Tensor,
        enable_pdl: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_q, sf)`` in the :func:`flashinfer.gemm.grouped_gemm_nt_masked` layout."""
        return _mxfp8_experts_quantize(input, mask, True, enable_pdl)

    @register_fake_op("flashinfer::silu_and_mul_mxfp8_experts_quantize_sm100")
    def _fake_silu_and_mul_mxfp8_experts_quantize_sm100(
        input: torch.Tensor,
        mask: torch.Tensor,
        enable_pdl: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _to_masked_gemm_layout(*_mxfp8_experts_outputs(input, True))

    @register_custom_op(
        "flashinfer::mxfp8_grouped_quant_sm100",
        mutates_args=(),
    )
    def mxfp8_grouped_quant_sm100(
        input: torch.Tensor,
        mask: torch.Tensor,
        enable_pdl: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_q, sf)`` in the :func:`flashinfer.gemm.grouped_gemm_nt_masked` layout."""
        return _mxfp8_experts_quantize(input, mask, False, enable_pdl)

    @register_fake_op("flashinfer::mxfp8_grouped_quant_sm100")
    def _fake_mxfp8_grouped_quant_sm100(
        input: torch.Tensor,
        mask: torch.Tensor,
        enable_pdl: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _to_masked_gemm_layout(*_mxfp8_experts_outputs(input, False))

    # Register the module
    return SimpleNamespace(
        mxfp8_quantize_sm100=mxfp8_quantize_sm100,
        mxfp8_dequantize_host_sm100=mxfp8_dequantize_host_sm100,
        silu_and_mul_mxfp8_experts_quantize_sm100=silu_and_mul_mxfp8_experts_quantize_sm100,
        mxfp8_grouped_quant_sm100=mxfp8_grouped_quant_sm100,
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
        The ``"cute-dsl"`` backend additionally supports fp32 input; the
        ``"cuda"`` backend does not.
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


# Generated Cake programs of the grouped MXFP8 quantizer: six programs per
# input dtype, all writing the same bytes.  Every program reads two-byte
# elements through one bfloat16 pointer (the float16 programs widen with their
# own conversion), so the host selects the program by dtype and shape and
# passes float16 storage as a bfloat16 view.  Each program ships for exact
# compute capability 10.0 (sm_100a) and 10.3 (sm_103a) only.
#
# The row-2D kernel launches one CTA per row and K tile; each CTA reads its
# group's mask word, then its row's blocks.
_CAKE_GROUPED_MXFP8_PROGRAMS: Dict[torch.dtype, CakeProgram] = {
    torch.bfloat16: _CAKE_GROUPED_MXFP8_BF16,
    torch.float16: _CAKE_GROUPED_MXFP8_F16,
}
# The bulk kernel has the row-2D grid and copies the CTA's blocks to shared
# memory before it reads the mask word, so the two round trips overlap.  It
# moves masked-out rows too, so it serves only problems small enough to be
# bound by the round trips rather than by bandwidth.
_CAKE_GROUPED_MXFP8_BULK_PROGRAMS: Dict[torch.dtype, CakeProgram] = {
    torch.bfloat16: _CAKE_GROUPED_MXFP8_BULK_BF16,
    torch.float16: _CAKE_GROUPED_MXFP8_BULK_F16,
}
# The tiled kernel gives each thread a block of ``ROWS`` rows of its group and
# reads the mask word once for them, so a sparse mask launches a fraction of
# the CTAs that would only read the mask word.  A row of fewer blocks than the
# CTA has threads shares the CTA with ``SUBROWS`` rows side by side.
_CAKE_GROUPED_MXFP8_TILED_PROGRAMS: Dict[torch.dtype, CakeProgram] = {
    torch.bfloat16: _CAKE_GROUPED_MXFP8_TILED_BF16,
    torch.float16: _CAKE_GROUPED_MXFP8_TILED_F16,
}
# The tiled programs below are builds of the same kernel for one K each.  They
# write the same bytes; they differ in the CTA width and in how a thread holds
# a block, computes its scale byte and forms its addresses.
#
# K = 7168: 224-thread CTAs, so a row of 224 blocks is one CTA instead of two
# and the grid has half the CTAs; the scale byte from integer operations.
_CAKE_GROUPED_MXFP8_TILED_WIDE_PROGRAMS: Dict[torch.dtype, CakeProgram] = {
    torch.bfloat16: _CAKE_GROUPED_MXFP8_TILED_WIDE_BF16,
    torch.float16: _CAKE_GROUPED_MXFP8_TILED_WIDE_F16,
}
# K = 4096: 256-thread CTAs, each holding two rows of 128 blocks side by side;
# packed two-byte pairs, the integer scale byte, one row address stepped per
# row and a launch bound of five CTAs per SM.
_CAKE_GROUPED_MXFP8_TILED_K4096_PROGRAMS: Dict[torch.dtype, CakeProgram] = {
    torch.bfloat16: _CAKE_GROUPED_MXFP8_TILED_K4096_BF16,
    torch.float16: _CAKE_GROUPED_MXFP8_TILED_K4096_F16,
}
# K = 2048: 128-thread CTAs with the packed pairs, the integer scale byte and
# the stepped row address.
_CAKE_GROUPED_MXFP8_TILED_K2048_PROGRAMS: Dict[torch.dtype, CakeProgram] = {
    torch.bfloat16: _CAKE_GROUPED_MXFP8_TILED_K2048_BF16,
    torch.float16: _CAKE_GROUPED_MXFP8_TILED_K2048_F16,
}
# Problems of at most this many input bytes take the bulk kernel.
_CAKE_BULK_MAX_INPUT_BYTES = 4 << 20
# Problems of at least this many rows (B * M) take a tiled kernel, at
# _CAKE_TILED_ROWS rows per thread: the one _CAKE_TILED_PROGRAMS_BY_K names
# for K, else the 128-thread one.
_CAKE_TILED_MIN_ROWS = 16384
_CAKE_TILED_ROWS = 2
_CAKE_TILED_PROGRAMS_BY_K: Dict[int, Dict[torch.dtype, CakeProgram]] = {
    7168: _CAKE_GROUPED_MXFP8_TILED_WIDE_PROGRAMS,
    4096: _CAKE_GROUPED_MXFP8_TILED_K4096_PROGRAMS,
    2048: _CAKE_GROUPED_MXFP8_TILED_K2048_PROGRAMS,
}
_CAKE_QUANT_BLOCK = 32
# One block row per M and one grid slice per B: CUDA caps grid.y and grid.z at 65535.
_CAKE_MAX_GRID_YZ = 65535
_CAKE_SCALE_TILE = 128


def _cake_grouped_mxfp8_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(
            f"the Cake grouped MXFP8 backend requires a CUDA device, got {device}"
        )
    return torch.cuda.current_device() if device.index is None else device.index


@functools.cache
def _cake_grouped_mxfp8_capability(device_index: int) -> Tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(device_index)
    return int(major), int(minor)


def _cake_grouped_mxfp8_target(program: CakeProgram, device: torch.device) -> str:
    """The exact compile target of ``program`` on ``device`` (capability cached per device index)."""
    major, minor = _cake_grouped_mxfp8_capability(
        _cake_grouped_mxfp8_device_index(device)
    )
    target = program.target_for((major, minor))
    if target is None:
        raise RuntimeError(
            f"the Cake grouped MXFP8 backend requires exact compute capability 10.0 or 10.3, got {major}.{minor}"
        )
    return target


def is_cake_grouped_mxfp8_quantize_available(
    dtype: torch.dtype, device: torch.device
) -> bool:
    """Whether a generated program serves ``dtype`` on ``device`` (no I/O; the capability query is cached)."""
    program = _CAKE_GROUPED_MXFP8_PROGRAMS.get(dtype)
    if program is None or device.type != "cuda":
        return False
    try:
        _cake_grouped_mxfp8_target(program, device)
    except RuntimeError:
        return False
    return True


def cake_grouped_mxfp8_quantize_program(
    dtype: torch.dtype, b: int, m: int, k: int, padded_k: Optional[int] = None
) -> CakeProgram:
    """The program that quantizes ``b`` groups of ``m`` rows of ``k`` elements of
    ``dtype`` into rows of ``padded_k`` codes (default ``round_up(k, 128)``),
    chosen from the shape alone (the host never reads the mask).

    The bulk and tiled programs take rows padded to ``round_up(k, 128)``; a wider
    padded row keeps the row-2D program, whose CTAs past ``k`` read nothing.
    """
    programs = (
        _CAKE_GROUPED_MXFP8_PROGRAMS,
        _CAKE_GROUPED_MXFP8_BULK_PROGRAMS,
        _CAKE_TILED_PROGRAMS_BY_K.get(k, _CAKE_GROUPED_MXFP8_TILED_PROGRAMS),
    )
    if dtype not in _CAKE_GROUPED_MXFP8_PROGRAMS:
        raise TypeError(f"unsupported Cake grouped MXFP8 input dtype: {dtype}")
    row2d, bulk, tiled = (table[dtype] for table in programs)
    if k < _CAKE_QUANT_BLOCK:
        return row2d
    if padded_k is not None and padded_k != _round_up(k, _CAKE_SCALE_TILE):
        return row2d
    if b * m * k * dtype.itemsize <= _CAKE_BULK_MAX_INPUT_BYTES:
        return bulk
    if b * m >= _CAKE_TILED_MIN_ROWS:
        return tiled
    return row2d


def _cake_grouped_mxfp8_subrows(program: CakeProgram, blocks_per_row: int) -> int:
    """Rows a CTA of the tiled ``program`` holds side by side: as many as its
    threads cover when a row has fewer blocks than the CTA has threads."""
    threads = program.block[0]
    return threads // blocks_per_row if blocks_per_row < threads else 1


def cake_grouped_mxfp8_quantize_grid(
    program: CakeProgram, b: int, m: int, blocks_per_row: int
) -> Tuple[int, int, int]:
    """Launch grid ``(grid_x, grid_y, grid_z)`` of ``program`` for ``b`` groups of
    ``m`` rows with ``blocks_per_row`` quantization blocks per row.

    One thread per block along x in CTAs of ``program.block[0]`` threads (the
    last CTA of a row is partial when ``blocks_per_row`` is not a multiple of
    the CTA width) and one grid slice per group; one CTA row per M, or, for a
    tiled program (one that takes ``ROWS``), one per ``ROWS * SUBROWS`` rows.
    """
    threads = program.block[0]
    grid_x = (blocks_per_row + threads - 1) // threads
    if "ROWS" not in program.args:
        return (grid_x, m, b)
    rows = _CAKE_TILED_ROWS * _cake_grouped_mxfp8_subrows(program, blocks_per_row)
    return (grid_x, (m + rows - 1) // rows, b)


def cake_grouped_mxfp8_quantize_launch(
    a: torch.Tensor,
    mask: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    """Quantize ``a`` ``[B, M, K]`` into the caller-owned ``quantized``
    ``[B, M, padded_K]`` (E4M3) and ``scales`` ``[B, padded_M, padded_K / 32]``
    (UE8M0 bytes in the 128x4 grouped-GEMM layout) on the current stream.

    No allocation, no synchronization; the generated binding checks the tensor
    dtypes, devices and contiguity and the host checks the shapes here, so
    callers pass contiguous tensors of exactly these shapes.  Empty problems
    (``B == 0`` or ``M == 0``) write nothing.
    """
    b, m, k = a.shape
    if b == 0 or m == 0:
        return
    if k % _CAKE_QUANT_BLOCK != 0:
        raise ValueError(f"K must be divisible by {_CAKE_QUANT_BLOCK}, got {k}")
    if m > _CAKE_MAX_GRID_YZ or b > _CAKE_MAX_GRID_YZ:
        raise ValueError(
            f"B and M must not exceed the CUDA grid limit {_CAKE_MAX_GRID_YZ}, got B={b}, M={m}"
        )
    # The binding validates dtype, device, contiguity and scalar ranges; the
    # shapes the kernel's scalars are derived from are validated here.
    quantized_shape = tuple(quantized.shape)
    scales_shape = tuple(scales.shape)
    padded_k = quantized_shape[2] if len(quantized_shape) == 3 else 0
    padded_m = scales_shape[1] if len(scales_shape) == 3 else 0
    if tuple(mask.shape) != (b,):
        raise ValueError(f"mask must have shape [B] = [{b}], got {list(mask.shape)}")
    if (
        quantized_shape != (b, m, padded_k)
        or padded_k % _CAKE_SCALE_TILE
        or padded_k < max(k, _CAKE_SCALE_TILE)
    ):
        raise ValueError(
            "quantized must have shape [B, M, padded_K] with padded_K a positive "
            f"multiple of {_CAKE_SCALE_TILE} and >= K={k}, got {list(quantized_shape)}"
        )
    blocks_per_row = padded_k // _CAKE_QUANT_BLOCK
    if (
        scales_shape != (b, padded_m, blocks_per_row)
        or padded_m % _CAKE_SCALE_TILE
        or padded_m < m
    ):
        raise ValueError(
            "scales must have shape [B, padded_M, padded_K / 32] with padded_M a "
            f"multiple of {_CAKE_SCALE_TILE} and >= M={m}, got {list(scales_shape)}"
        )
    program = cake_grouped_mxfp8_quantize_program(a.dtype, b, m, k, padded_k)
    target = _cake_grouped_mxfp8_target(program, a.device)
    pm_tiles = padded_m // _CAKE_SCALE_TILE
    pk_tiles = padded_k // _CAKE_SCALE_TILE
    grid_x, grid_y, grid_z = cake_grouped_mxfp8_quantize_grid(
        program, b, m, blocks_per_row
    )
    values = dict(
        x=a if a.dtype == torch.bfloat16 else a.view(torch.bfloat16),
        mask=mask,
        quantized=quantized,
        scales=scales.view(b, pm_tiles, pk_tiles, 32, 4, 4),
        M=m,
        K=k,
        PADDED_K=padded_k,
        PM_TILES=pm_tiles,
        PK_TILES=pk_tiles,
        BLOCKS_PER_ROW=blocks_per_row,
        TOTAL_TASKS=b * m * blocks_per_row,
        ROWS=_CAKE_TILED_ROWS,
        SUBROWS=_cake_grouped_mxfp8_subrows(program, blocks_per_row),
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
    )
    entry = program.entry(target)
    arguments = [values[name] for name in program.args]
    with tvm_ffi.use_torch_stream():
        entry(*arguments)


@functools.cache
def get_cake_mxfp8_grouped_quantization_module():
    """Register the generated Cake grouped MXFP8 op and fake metadata twin."""

    @register_custom_op(
        "flashinfer::cake_mxfp8_grouped_quantize",
        mutates_args=(),
    )
    def cake_mxfp8_grouped_quantize_impl(
        a: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        cake_grouped_mxfp8_quantize_launch(a, mask, output, output_scales)

        output = output.permute(1, 2, 0)
        output_scales = output_scales.view(b, padded_m // 128, scale_k // 4, 32, 4, 4)
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return output, output_scales

    @register_fake_op("flashinfer::cake_mxfp8_grouped_quantize")
    def _fake_cake_mxfp8_grouped_quantize(
        a: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_mxfp8_grouped_quantization_module()._fake_mxfp8_grouped_quantize(
            a, mask
        )

    return SimpleNamespace(
        cake_mxfp8_grouped_quantize_impl=cake_mxfp8_grouped_quantize_impl,
        _fake_cake_mxfp8_grouped_quantize=_fake_cake_mxfp8_grouped_quantize,
    )


def _mxfp8_grouped_quantize_cake(
    a: torch.Tensor,
    mask: torch.Tensor,
    *,
    backend: Literal["cake"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch an already validated request to the generated Cake program."""

    if backend != "cake":
        raise ValueError(f"internal Cake dispatch received backend={backend!r}")
    if not is_cake_grouped_mxfp8_quantize_available(a.dtype, a.device):
        raise RuntimeError(
            "backend='cake' requires float16 or bfloat16 input on a device of "
            f"exact compute capability 10.0 or 10.3; got dtype={a.dtype} on "
            f"{torch.cuda.get_device_name(a.device)}"
        )
    if not a.is_contiguous() or not mask.is_contiguous():
        raise ValueError(
            "backend='cake' requires contiguous a [B, M, K] and mask [B]; pass "
            "a.contiguous() / mask.contiguous() explicitly instead of a view"
        )
    return (
        get_cake_mxfp8_grouped_quantization_module().cake_mxfp8_grouped_quantize_impl(
            a, mask
        )
    )


@flashinfer_api(trace=mxfp8_grouped_quantize_trace)
def mxfp8_grouped_quantize(
    a: torch.Tensor,
    mask: torch.Tensor,
    backend: Literal["cutile", "cake", "cuda"] = "cutile",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize grouped inputs to MXFP8 with UE8M0 block scales.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor of shape ``[B, M, K]`` with dtype ``float16`` or
        ``bfloat16``.
    mask : torch.Tensor
        Int32 CUDA tensor of shape ``[B]`` with the number of valid rows of
        each group. It is read only on the device, so its values are not
        validated. The ``"cuda"`` backend uses the first ``min(mask[i], M)``
        rows. The ``"cutile"`` and ``"cake"`` backends require
        ``0 <= mask[i] <= M``; with ``mask[i] > M`` the cuTile kernel writes
        out of bounds.
    backend : {``"cutile"``, ``"cake"``, ``"cuda"``}
        Implementation backend. ``"cutile"`` (default) requires the
        ``cuda-tile`` package and ``"cake"`` a generated profile for the exact
        Blackwell architecture. ``"cake"`` and ``"cuda"`` require contiguous
        ``a`` and ``mask``; ``"cuda"`` also requires ``a`` to be 32-byte
        aligned. A failed launch is not retried on another backend.

        Outputs match across backends for finite inputs, including bfloat16
        subnormals. A zero UE8M0 scale byte produces signed-zero payloads.
        ``"cuda"`` quantizes NaN and infinities to 448, as
        :func:`mxfp8_quantize` does; blocks that hold them can differ.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(x_q, sf)`` where ``x_q`` has logical shape
        ``[M, padded_K, B]`` with dtype ``float8_e4m3fn`` and ``sf`` has
        logical shape ``[32, 4, padded_M // 128, 4, padded_K // 128, B]``
        with dtype ``uint8``. ``padded_K`` rounds ``K`` up to a multiple
        of 128 and ``padded_M`` rounds ``M`` up to a multiple of 128. The
        physical layouts are grouped by ``B`` and then permuted to match
        FlashInfer masked grouped GEMM conventions.

        Rows past the valid ones (and their scale factors) are unspecified.
    """

    if backend not in ("cutile", "cake", "cuda"):
        raise ValueError(f"Unsupported backend for mxfp8_grouped_quantize: {backend!r}")
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

    if a.size(2) % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {a.size(2)}")

    major, _ = get_compute_capability(a.device)
    if major < 10:
        raise RuntimeError("mxfp8_grouped_quantize requires SM100 or newer")

    if backend == "cuda":
        if not a.is_contiguous() or not mask.is_contiguous():
            raise ValueError(
                "backend='cuda' requires contiguous a [B, M, K] and mask [B]; pass "
                "a.contiguous() / mask.contiguous() explicitly instead of a view"
            )
        if a.data_ptr() % _MXFP8_EXPERTS_INPUT_ALIGNMENT != 0:
            raise ValueError(
                f"backend='cuda' requires a {_MXFP8_EXPERTS_INPUT_ALIGNMENT}-byte "
                "aligned a; copy an offset view with a.clone()"
            )
        return get_mxfp8_quantization_sm100_module().mxfp8_grouped_quant_sm100(
            a, mask, device_support_pdl(a.device)
        )

    if backend == "cutile":
        if not is_cuda_tile_available():
            raise RuntimeError(
                "mxfp8_grouped_quantize requires the cuTile backend; install "
                "cuda-tile>=1.4.0 and ensure the tileiras compiler is available."
            )
        return get_mxfp8_grouped_quantization_module().mxfp8_grouped_quantize_impl(
            a, mask
        )

    return _mxfp8_grouped_quantize_cake(a, mask, backend="cake")


@flashinfer_api(trace=silu_and_mul_mxfp8_experts_quantize_trace)
def silu_and_mul_mxfp8_experts_quantize(
    a: torch.Tensor,
    mask: torch.Tensor,
    *,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused SiLU + mul + MXFP8 quantization with a per-expert row mask.

    Quantizes ``silu(a[..., :K]) * a[..., K:]``, rounded to the input dtype,
    to MXFP8 (E4M3 values, one UE8M0 scale per 32 values). Requires SM100+.
    The outputs are the ``(A, SFA)`` operand of the second
    :func:`flashinfer.gemm.grouped_gemm_nt_masked` of a MoE.

    Parameters
    ----------
    a : torch.Tensor
        Contiguous, 32-byte aligned ``float16`` or ``bfloat16`` tensor of
        shape ``[B, M, 2 * K]``; each row holds the gate values, then the up
        values. ``K`` must be divisible by 32.
    mask : torch.Tensor
        Contiguous int32 CUDA tensor of shape ``[B]``, read on the device.
        Only the first ``min(mask[i], M)`` rows of expert ``i`` are read and
        written.
    enable_pdl : bool, optional
        Whether to enable Programmatic Dependent Launch. Auto-detected from
        the device capability when ``None``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(x_q, sf)`` in the layout of :func:`mxfp8_grouped_quantize`: ``x_q``
        has logical shape ``[M, padded_K, B]`` with dtype ``float8_e4m3fn``
        and ``sf`` has logical shape
        ``[32, 4, padded_M // 128, 4, padded_K // 128, B]`` with dtype
        ``uint8``, where ``padded_K`` and ``padded_M`` round ``K`` and ``M`` up
        to multiples of 128. Rows past the valid ones (and their scale
        factors) are unspecified.

        NaN and infinities quantize to 448, and the other values of a block
        with an infinity to zero.
    """
    if a.dim() != 3:
        raise ValueError("a must be a 3D tensor with shape [B, M, 2 * K]")
    if not a.is_cuda:
        raise ValueError("a must be a CUDA tensor")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a dtype must be torch.float16 or torch.bfloat16")
    if not a.is_contiguous():
        raise ValueError("a must be contiguous")
    if a.data_ptr() % _MXFP8_EXPERTS_INPUT_ALIGNMENT != 0:
        raise ValueError(
            f"a must be {_MXFP8_EXPERTS_INPUT_ALIGNMENT}-byte aligned; copy an offset view "
            "with a.clone()"
        )
    if a.size(2) % 64 != 0:
        raise ValueError(
            f"a last dim must be 2 * K with K divisible by 32, got {a.size(2)}"
        )
    if mask.dim() != 1 or mask.size(0) != a.size(0):
        raise ValueError("mask must be a 1D tensor with one entry per expert")
    if mask.dtype != torch.int32:
        raise ValueError("mask dtype must be torch.int32")
    if mask.device != a.device:
        raise ValueError("mask must live on the same CUDA device as a")
    if not mask.is_contiguous():
        raise ValueError("mask must be contiguous")

    major, _ = get_compute_capability(a.device)
    if major < 10:
        raise RuntimeError(
            "silu_and_mul_mxfp8_experts_quantize requires SM100 or newer"
        )
    if enable_pdl is None:
        enable_pdl = device_support_pdl(a.device)

    return (
        get_mxfp8_quantization_sm100_module().silu_and_mul_mxfp8_experts_quantize_sm100(
            a, mask, enable_pdl
        )
    )


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


@flashinfer_api
def per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dst_dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    backend: Literal["cutile"] = "cutile",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-token group 8-bit quantization (FP8 or INT8).

    Quantizes ``x`` along its last dimension in contiguous groups of
    ``group_size`` elements, producing a quantized tensor and a per-group scale
    tensor. This is the per-token-group quantization used by block-scaled FP8 /
    INT8 GEMM paths (e.g. DeepGEMM-style grouped quantization).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to quantize; the last dimension must be a multiple of
        ``group_size``.
    group_size : int
        Number of elements per quantization group (along the last dimension).
    eps : float
        Epsilon for numerical stability when computing per-group scales.
    dst_dtype : Optional[torch.dtype]
        Quantized output dtype (``torch.float8_e4m3fn`` or ``torch.int8``).
        Defaults to ``torch.float8_e4m3fn``.
    column_major_scales : bool
        If ``True``, return the scale tensor in column-major (Fortran) memory
        layout. The logical shape is unchanged.
    scale_tma_aligned : bool
        If ``True``, pad the scale tensor's leading dimension for TMA alignment.
    scale_ue8m0 : bool
        If ``True``, encode scales in the UE8M0 format (Blackwell / sm100+).
    backend : str
        Implementation backend. Currently only ``"cutile"`` (the cuda.tile
        Python backend) is supported.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(x_q, x_s)``: the quantized tensor (same shape as ``x``) and the
        per-group scale tensor of shape ``(*x.shape[:-1], x.shape[-1] // group_size)``.
    """
    if backend == "cutile":
        from .kernels.cutile.per_token_group_quant_8bit_cutile import (
            per_token_group_quant_8bit_cutile,
        )

        return per_token_group_quant_8bit_cutile(
            x,
            group_size,
            eps=eps,
            dst_dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=scale_ue8m0,
        )

    raise ValueError(f"Unsupported backend for per_token_group_quant_8bit: {backend!r}")
