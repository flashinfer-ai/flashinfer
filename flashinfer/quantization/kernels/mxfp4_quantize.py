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

MXFP4 Quantization using CuTe-DSL
=================================

MXFP4 quantization kernel using CuTe-DSL.
Supports multiple scale factor layouts: swizzled 128x4 and linear.

Uses flat SF-block iteration (like MXFP8) for 100% thread utilization
at all K values. Each thread processes one SF block (32 elements) from
a global flat pool, with row_idx and col_idx derived via integer division.

"""

import functools
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint8

from ...api_logging import flashinfer_api
from ...cute_dsl.fp4_common import get_ptr_as_int64, st_global_u64
from ...cute_dsl.utils import get_num_sm
from ..quantization_cute_dsl_utils import (
    # MXFP4 Constants
    MXFP4_SF_VEC_SIZE,
    WARP_SIZE,
    ROW_TILE_SIZE,
    # Low-level intrinsics
    compute_sf_index_swizzled_128x4_gpu,
    compute_sf_index_linear_gpu,
    # High-level helpers (MXFP4)
    process_mxfp4_block_half,
    process_mxfp4_block_bfloat,
)

SF_LAYOUT_128x4 = 0
SF_LAYOUT_LINEAR = 2


# Blocks per SM for occupancy target
_BLOCKS_PER_SM = 4

# Maximum threads per block
_MAX_THREADS_PER_BLOCK = 1024

# Flat iteration: fixed 16 warps (512 threads), 1 SF block per thread
_WARPS_PER_BLOCK = 16
_SF_BLOCKS_PER_TB = _WARPS_PER_BLOCK * WARP_SIZE  # 512 SF blocks per thread block


def _compute_swizzled_layout_sf_size(
    total_row: int, total_column: int, row_size: int = 128
) -> int:
    """Compute size of swizzled scale factor buffer."""
    padded_row = (total_row + row_size - 1) // row_size * row_size
    padded_column = (total_column + 3) // 4 * 4
    return padded_row * padded_column


# =============================================================================
# CuTe-DSL Kernel Class for MXFP4 — Flat SF-Block Iteration
# =============================================================================


class MXFP4QuantizeKernel:
    """
    MXFP4 quantization kernel with flat SF-block iteration.

    Supported layouts:
    - 128x4 (swizzled): Optimized for GEMM with large tileN
    - linear: Simple row-major layout, no swizzling

    Uses flat SF-block iteration (like MXFP8) instead of row-based iteration.
    This gives 100% thread utilization at all K values, eliminating the
    K-dependent thread waste that row-based iteration suffers from.

    Each thread processes one SF block (32 elements):
    - Loads 32 fp16/bf16 elements (4 × 128-bit loads)
    - Computes max absolute value using SIMD reduction
    - Computes UE8M0 scale factor
    - Converts to E2M1 (4-bit) format
    - Stores 16 bytes packed output + scale factor (layout-specific)

    SF blocks are assigned from a flat global pool via grid-stride loop.
    row_idx and col_idx are derived from the flat SF index via integer
    division, enabling layout-specific scale factor writes.

    This kernel is M-agnostic: compiled once per (K, dtype, sf_layout, pdl)
    combination.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        K: int,
        sf_layout: int = SF_LAYOUT_128x4,
        enable_pdl: bool = False,
    ):
        self.dtype = dtype
        self.K = K
        self.is_bfloat16 = dtype == cutlass.BFloat16
        self.enable_pdl = enable_pdl
        self.sf_layout = sf_layout
        self.sf_is_128x4 = sf_layout == SF_LAYOUT_128x4

        assert K % MXFP4_SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // MXFP4_SF_VEC_SIZE

        if sf_layout == SF_LAYOUT_LINEAR:
            self.padded_sf_cols = self.num_sf_blocks_per_row
            self.row_tile_size = 1
        else:
            self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4
            self.row_tile_size = ROW_TILE_SIZE  # 128

    @cute.jit
    def _compute_sf_offset(
        self, row_idx: Int32, col_idx: Int32, padded_cols: Int32
    ) -> Int32:
        """Compute scale factor offset based on layout (compile-time dispatch)."""
        if cutlass.const_expr(self.sf_is_128x4):
            return compute_sf_index_swizzled_128x4_gpu(row_idx, col_idx, padded_cols)
        else:
            return compute_sf_index_linear_gpu(row_idx, col_idx, padded_cols)

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        M: Int32,
        padded_M: Int32,
        total_sf_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(mInput, mOutput, mScales, M, padded_M, total_sf_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[_WARPS_PER_BLOCK * WARP_SIZE, 1, 1],
            max_number_threads=[_MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=_BLOCKS_PER_SM,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        M: Int32,
        padded_M: Int32,
        total_sf_blocks: Int32,
    ):
        """
        MXFP4 quantization with flat SF-block iteration.

        Processes SF blocks from a global flat pool. Each thread handles
        one SF block (32 elements). Row and column indices are derived
        from the flat SF index.

        Padding rows (row_idx >= M but < padded_M) and padding SF columns
        (col_idx >= num_sf_blocks_per_row but < padded_sf_cols) need their
        scale factors zeroed. These are handled in separate passes after
        the main quantization loop.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        padded_sf_cols = self.padded_sf_cols
        sf_blocks_per_tb = _SF_BLOCKS_PER_TB
        stride = grid_dim_x * sf_blocks_per_tb

        # ===== Main quantization loop: flat SF-block iteration =====
        sf_idx = bidx * sf_blocks_per_tb + tidx

        while sf_idx < total_sf_blocks:
            row_idx = sf_idx // num_sf_blocks_per_row
            col_idx = sf_idx % num_sf_blocks_per_row

            elem_base = col_idx * MXFP4_SF_VEC_SIZE
            row_input = mInput[row_idx, None]

            # Process block: load, compute scale, convert to E2M1
            if cutlass.const_expr(self.is_bfloat16):
                (
                    _,
                    scale_ue8m0,
                    packed64_0,
                    packed64_1,
                ) = process_mxfp4_block_bfloat(row_input, elem_base)
            else:
                (
                    _,
                    scale_ue8m0,
                    packed64_0,
                    packed64_1,
                ) = process_mxfp4_block_half(row_input, elem_base)

            # Write scale factor using layout-specific indexing
            sf_offset = self._compute_sf_offset(
                row_idx, col_idx, padded_sf_cols
            )
            mScales[sf_offset] = scale_ue8m0

            # Store 16 bytes (32 FP4 values = 2 × st.global.u64)
            row_output = mOutput[row_idx, None]
            out_base = col_idx * (MXFP4_SF_VEC_SIZE // 2)
            out_ptr0 = get_ptr_as_int64(row_output, out_base)
            out_ptr1 = get_ptr_as_int64(row_output, out_base + Int32(8))
            st_global_u64(out_ptr0, packed64_0)
            st_global_u64(out_ptr1, packed64_1)

            sf_idx = sf_idx + stride

        # ===== Padding pass: zero SF entries for padding rows =====
        # Padding rows are rows in [M, padded_M) that need their scale
        # factors zeroed in the layout. Only applies when padded_M > M
        # (swizzled layout pads to multiples of 128).
        padding_sf_start = M * padded_sf_cols
        total_padding_sf = padded_M * padded_sf_cols
        padding_sf_idx = padding_sf_start + bidx * sf_blocks_per_tb + tidx

        while padding_sf_idx < total_padding_sf:
            padding_row = padding_sf_idx // padded_sf_cols
            padding_col = padding_sf_idx % padded_sf_cols
            sf_offset = self._compute_sf_offset(
                padding_row, padding_col, padded_sf_cols
            )
            mScales[sf_offset] = Uint8(0)
            padding_sf_idx = padding_sf_idx + stride

        # ===== Padding pass: zero SF entries for padding columns =====
        # Only needed when num_sf_blocks_per_row < padded_sf_cols
        # (swizzled layout pads SF columns to multiples of 4).
        if cutlass.const_expr(
            self.num_sf_blocks_per_row != self.padded_sf_cols
        ):
            padding_cols_per_row = padded_sf_cols - num_sf_blocks_per_row
            total_col_padding = M * padding_cols_per_row
            col_pad_idx = bidx * sf_blocks_per_tb + tidx

            while col_pad_idx < total_col_padding:
                row_for_pad = col_pad_idx // padding_cols_per_row
                col_offset = col_pad_idx % padding_cols_per_row
                actual_col = num_sf_blocks_per_row + col_offset
                sf_offset = self._compute_sf_offset(
                    row_for_pad, actual_col, padded_sf_cols
                )
                mScales[sf_offset] = Uint8(0)
                col_pad_idx = col_pad_idx + stride

        # PDL: Signal that dependent kernels can start early
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# PyTorch Integration with TVM-FFI
# =============================================================================


@functools.cache
def _get_compiled_kernel_mxfp4(
    is_bfloat16: bool,
    K: int,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool = False,
) -> Tuple[Callable, int]:
    """
    Get or compile MXFP4 kernel with TVM-FFI.

    Cached by (K, dtype, sf_layout, pdl) - M-agnostic, device-independent
    compilation.

    Returns:
        Tuple of (compiled_kernel, sf_blocks_per_tb) where sf_blocks_per_tb
        is used by the caller to compute num_blocks at runtime.
    """
    cutlass_dtype = cutlass.BFloat16 if is_bfloat16 else cutlass.Float16
    kernel_obj = MXFP4QuantizeKernel(cutlass_dtype, K, sf_layout, enable_pdl)

    # Use symbolic M for dynamic batch sizes
    sym_m = cute.sym_int()

    # Create fake tensors for compilation
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K // 2), stride_order=(1, 0), assumed_align=16
    )
    sym_scale_size = cute.sym_int()
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_scale_size,), assumed_align=16
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        input_fake,
        output_fake,
        scales_fake,
        Int32(1),  # Dummy M
        Int32(128),  # Dummy padded_M
        Int32(1),  # Dummy total_sf_blocks
        Int32(1),  # Dummy num_blocks
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel, _SF_BLOCKS_PER_TB


@flashinfer_api
def mxfp4_quantize_cute_dsl(
    input: torch.Tensor,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to MXFP4 format using CuTe-DSL kernel.

    This is a GPU implementation matching FlashInfer's mxfp4_quantize() behavior:
    - Global scale computed as (448 * 6) / max(|input|)
    - UE8M0 scale factors
    - E2M1 output format (4-bit, 2 values per byte)
    - Supports 128x4 (swizzled) and linear scale factor layouts

    The kernel is compiled once per (K, dtype, sf_layout, pdl) combination and
    handles varying M (batch size) at runtime without recompilation.

    Args:
        input: Input tensor of shape [M, K] with dtype fp16/bf16
        sf_layout: Scale factor layout (0=128x4, 2=linear).
        enable_pdl: Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability (SM >= 9.0).

    Returns:
        Tuple of:
            - fp4_tensor: Quantized tensor of shape [M, K/2] with dtype uint8
            - scale_tensor: Scale factors as uint8 tensor
              reshaped to [padded_rows, K/32]
    """
    from ...utils import device_support_pdl

    _valid_sf_layouts = (SF_LAYOUT_128x4, SF_LAYOUT_LINEAR)
    assert sf_layout in _valid_sf_layouts, (
        f"sf_layout must be one of {_valid_sf_layouts}, got {sf_layout}"
    )
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"Input dtype must be float16 or bfloat16, got {input.dtype}"
    )
    assert input.is_cuda, "Input must be on CUDA device"

    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)

    if input.dim() > 2:
        m = input.numel() // input.shape[-1]
        k = input.shape[-1]
        input = input.reshape(m, k)
    else:
        m, k = input.shape

    assert k % MXFP4_SF_VEC_SIZE == 0, (
        f"K ({k}) must be divisible by MXFP4_SF_VEC_SIZE={MXFP4_SF_VEC_SIZE}"
    )

    input = input.contiguous()
    is_bfloat16 = input.dtype == torch.bfloat16

    target_grid = get_num_sm(input.device) * _BLOCKS_PER_SM

    num_sf_blocks_per_row = k // MXFP4_SF_VEC_SIZE

    if sf_layout == SF_LAYOUT_LINEAR:
        # NOTE: When adding a TMA-based kernel, padded_m must be rounded up to the
        # TMA tile row dimension (e.g. round_up(m, tma_tile_rows)) and scale_output
        # must be trimmed to m * num_sf_blocks_per_row before returning.
        # See PR f4d10d9 for the analogous CUDA fix.
        padded_m = m
        padded_sf_cols = num_sf_blocks_per_row
    else:
        padded_m = ((m + ROW_TILE_SIZE - 1) // ROW_TILE_SIZE) * ROW_TILE_SIZE
        padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4

    scale_output_size = padded_m * padded_sf_cols

    # Total SF blocks for actual data (not padding)
    total_sf_blocks = m * num_sf_blocks_per_row

    # Get or compile kernel (device-independent)
    kernel_fn, sf_blocks_per_tb = _get_compiled_kernel_mxfp4(
        is_bfloat16, k, sf_layout, enable_pdl
    )

    # Compute grid size from flat SF block count
    num_blocks = min(
        (total_sf_blocks + sf_blocks_per_tb - 1) // sf_blocks_per_tb,
        target_grid,
    )

    fp4_output = torch.empty(m, k // 2, dtype=torch.uint8, device=input.device)
    scale_output = torch.empty(
        scale_output_size, dtype=torch.uint8, device=input.device
    )

    kernel_fn(
        input, fp4_output, scale_output, m, padded_m, total_sf_blocks, num_blocks
    )

    scale_output = scale_output.reshape(-1, num_sf_blocks_per_row)

    return fp4_output, scale_output


__all__ = [
    "SF_LAYOUT_128x4",
    "SF_LAYOUT_LINEAR",
    "MXFP4QuantizeKernel",
    "mxfp4_quantize_cute_dsl",
    "_get_compiled_kernel_mxfp4",
]
