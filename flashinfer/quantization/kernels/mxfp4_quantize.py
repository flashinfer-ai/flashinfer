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

Dual-path optimization following the MXFP8 pattern:
- Linear layout: flat SF-block iteration for 100% thread utilization
- Swizzled layout: row-based iteration with multi-row / column-loop paths
"""

import functools
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint8

from ...api_logging import flashinfer_api
from ...cute_dsl.fp4_common import (
    get_ptr_as_int64,
    st_global_u64,
)
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

# Thread count bounds for swizzled kernel
_MIN_THREADS = 128
_MAX_THREADS = 512

# Linear kernel: fixed 16 warps (512 threads), 1 SF block per thread
_LINEAR_WARPS_PER_BLOCK = 16
_LINEAR_SF_BLOCKS_PER_TB = _LINEAR_WARPS_PER_BLOCK * WARP_SIZE  # 512


def _compute_swizzled_layout_sf_size(
    total_row: int, total_column: int, row_size: int = 128
) -> int:
    """Compute size of swizzled scale factor buffer."""
    padded_row = (total_row + row_size - 1) // row_size * row_size
    padded_column = (total_column + 3) // 4 * 4
    return padded_row * padded_column


def _compute_optimal_threads(K: int) -> int:
    """
    Compute optimal thread count for 100% utilization in the swizzled kernel.

    For MXFP4, each thread processes 1 SF block (32 elements), so:
        threads_per_row = K / 32

    We want num_threads to be a multiple of threads_per_row so that
    rows_per_block = num_threads / threads_per_row is an integer.

    We prefer LARGER thread counts (up to _MAX_THREADS) for better occupancy.

    If threads_per_row > _MAX_THREADS, we use _MAX_THREADS with a column loop.

    Args:
        K: Number of columns (must be divisible by 32)

    Returns:
        Optimal number of threads per block
    """
    threads_per_row = K // MXFP4_SF_VEC_SIZE  # K / 32

    if threads_per_row > _MAX_THREADS:
        # Column loop mode: use maximum threads
        return _MAX_THREADS

    # Find largest multiple of threads_per_row in [_MIN_THREADS, _MAX_THREADS]
    largest = (_MAX_THREADS // threads_per_row) * threads_per_row
    if largest >= _MIN_THREADS:
        return largest

    # If largest multiple is below _MIN_THREADS, use smallest valid one
    candidate = threads_per_row
    while candidate < _MIN_THREADS:
        candidate += threads_per_row
    if candidate <= _MAX_THREADS:
        return candidate

    # Fallback (shouldn't happen for reasonable K)
    return _MAX_THREADS


# =============================================================================
# CuTe-DSL Kernel Class for Linear Layout — Flat SF-Block Iteration
# =============================================================================


class MXFP4QuantizeLinearKernel:
    """
    MXFP4 quantization kernel optimized for LINEAR layout.

    Uses flat SF-block iteration for efficient memory access. Row and
    column indices are derived from the flat SF index via integer division.

    No padding passes are needed since for linear layout:
    - padded_m == m (no row padding)
    - padded_sf_cols == num_sf_blocks_per_row (no column padding)

    This kernel is M-agnostic: compiled once per (K, dtype, pdl) combination.
    Each thread handles one SF block (32 elements).
    """

    WARPS_PER_BLOCK = _LINEAR_WARPS_PER_BLOCK
    SF_BLOCKS_PER_TB = _LINEAR_SF_BLOCKS_PER_TB  # 512

    def __init__(
        self,
        dtype: cutlass.Numeric,
        K: int,
        enable_pdl: bool = False,
    ):
        self.dtype = dtype
        self.K = K
        self.is_bfloat16 = dtype == cutlass.BFloat16
        self.enable_pdl = enable_pdl

        assert K % MXFP4_SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // MXFP4_SF_VEC_SIZE

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        M: Int32,
        total_sf_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        threads_per_block = self.WARPS_PER_BLOCK * WARP_SIZE

        self.kernel(mInput, mOutput, mScales, M, total_sf_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[threads_per_block, 1, 1],
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
        total_sf_blocks: Int32,
    ):
        """
        MXFP4 quantization with flat SF-block iteration for linear layout.

        Each thread handles one SF block (32 elements).
        Row and column indices are derived from the flat SF index.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        sf_blocks_per_tb = self.SF_BLOCKS_PER_TB

        stride = grid_dim_x * sf_blocks_per_tb

        # Flat SF-block iteration
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

            # Write scale factor using linear indexing
            sf_offset = compute_sf_index_linear_gpu(
                row_idx, col_idx, num_sf_blocks_per_row
            )
            mScales[sf_offset] = scale_ue8m0

            # Store 16 bytes (32 FP4 values = 2 x st.global.u64)
            row_output = mOutput[row_idx, None]
            out_base = col_idx * (MXFP4_SF_VEC_SIZE // 2)
            out_ptr0 = get_ptr_as_int64(row_output, out_base)
            out_ptr1 = get_ptr_as_int64(row_output, out_base + Int32(8))
            st_global_u64(out_ptr0, packed64_0)
            st_global_u64(out_ptr1, packed64_1)

            sf_idx = sf_idx + stride

        # PDL: Signal that dependent kernels can start early
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# CuTe-DSL Kernel Class for Swizzled Layout — Row-Based Iteration
# =============================================================================


class MXFP4QuantizeSwizzledKernel:
    """
    MXFP4 quantization kernel optimized for SWIZZLED (128x4) layout.

    Key optimizations:
    - Multi-row processing: threads process multiple rows per block when K is small
    - Row-based iteration with grid-stride loop
    - Padding row fast path - only zero out scale factors

    Thread utilization optimization:
    - Dynamic thread count based on K for 100% thread utilization
    - For small K: Multiple rows processed per block iteration
    - For large K: Single row with column loop

    For MXFP4, each thread processes 1 SF block (32 elements) independently,
    so threads_per_row = num_sf_blocks_per_row = K/32.

    This kernel is M-agnostic: compiled once per (K, dtype, pdl) combination.
    M-dependent values (M, padded_M) are passed at runtime.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        K: int,
        enable_pdl: bool = False,
    ):
        self.dtype = dtype
        self.K = K
        self.is_bfloat16 = dtype == cutlass.BFloat16
        self.enable_pdl = enable_pdl

        assert K % MXFP4_SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // MXFP4_SF_VEC_SIZE
        self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4

        # Compute optimal thread count for 100% utilization
        self.num_threads = _compute_optimal_threads(K)
        self.threads_per_row = self.num_sf_blocks_per_row  # 1 thread per SF block

        # Multi-row processing constants (compile-time)
        if self.threads_per_row <= self.num_threads:
            self.rows_per_block = self.num_threads // self.threads_per_row
            self.needs_col_loop = False
        else:
            self.rows_per_block = 1
            self.needs_col_loop = True

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        M: Int32,
        padded_M: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(mInput, mOutput, mScales, M, padded_M).launch(
            grid=[num_blocks, 1, 1],
            block=[self.num_threads, 1, 1],
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
    ):
        """
        Row-based kernel for swizzled layout.

        When K is small: each block processes multiple rows simultaneously.
        When K is large: each block processes one row with column loop.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        # Compile-time constants
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        padded_sf_cols = self.padded_sf_cols
        threads_per_row = self.threads_per_row
        rows_per_block = self.rows_per_block

        if cutlass.const_expr(self.needs_col_loop):
            # Large K path: single row per block iteration with column loop
            # Each thread maps to one SF block; threads stride over columns
            num_threads = self.num_threads

            row_idx = bidx
            while row_idx < padded_M:
                is_padding_row = row_idx >= M

                if is_padding_row:
                    # Fast path: padding row - only zero out scale factors
                    sf_col_idx = tidx
                    while sf_col_idx < padded_sf_cols:
                        sf_offset = compute_sf_index_swizzled_128x4_gpu(
                            row_idx, sf_col_idx, padded_sf_cols
                        )
                        mScales[sf_offset] = Uint8(0)
                        sf_col_idx = sf_col_idx + num_threads
                else:
                    # Normal path: process actual data row with column loop
                    sf_col_idx = tidx
                    while sf_col_idx < num_sf_blocks_per_row:
                        elem_base = sf_col_idx * MXFP4_SF_VEC_SIZE
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

                        # Write scale factor using swizzled indexing
                        sf_offset = compute_sf_index_swizzled_128x4_gpu(
                            row_idx, sf_col_idx, padded_sf_cols
                        )
                        mScales[sf_offset] = scale_ue8m0

                        # Store 16 bytes (32 FP4 values = 2 x st.global.u64)
                        row_output = mOutput[row_idx, None]
                        out_base = sf_col_idx * (MXFP4_SF_VEC_SIZE // 2)
                        out_ptr0 = get_ptr_as_int64(row_output, out_base)
                        out_ptr1 = get_ptr_as_int64(row_output, out_base + Int32(8))
                        st_global_u64(out_ptr0, packed64_0)
                        st_global_u64(out_ptr1, packed64_1)

                        sf_col_idx = sf_col_idx + num_threads

                    # Handle padding columns for this row
                    sf_col_idx = num_sf_blocks_per_row + tidx
                    while sf_col_idx < padded_sf_cols:
                        sf_offset = compute_sf_index_swizzled_128x4_gpu(
                            row_idx, sf_col_idx, padded_sf_cols
                        )
                        mScales[sf_offset] = Uint8(0)
                        sf_col_idx = sf_col_idx + num_threads

                row_idx = row_idx + grid_dim_x
        else:
            # Small K path: multi-row processing
            # Thread mapping: tidx -> (row_in_block, sf_idx_in_row)
            row_in_block = tidx // threads_per_row
            sf_idx_in_row = tidx % threads_per_row

            # Grid-stride loop over row batches
            row_batch_idx = bidx
            # Initialize row_idx before while loop (CuTe DSL requires variables
            # modified in while loops to be defined before the loop)
            row_idx = row_batch_idx * rows_per_block + row_in_block
            while row_batch_idx * rows_per_block < padded_M:
                if row_idx < padded_M:
                    is_padding_row = row_idx >= M

                    if is_padding_row:
                        # Fast path: padding row - zero out ALL padded_sf_cols
                        # Thread-stride loop since padded_sf_cols may exceed
                        # threads_per_row (e.g. K=64: threads_per_row=2,
                        # padded_sf_cols=4)
                        local_sf_idx = sf_idx_in_row
                        while local_sf_idx < padded_sf_cols:
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, local_sf_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)
                            local_sf_idx = local_sf_idx + threads_per_row
                    else:
                        # Normal path: process actual data
                        if sf_idx_in_row < num_sf_blocks_per_row:
                            elem_base = sf_idx_in_row * MXFP4_SF_VEC_SIZE
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

                            # Write scale factor using swizzled indexing
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, sf_idx_in_row, padded_sf_cols
                            )
                            mScales[sf_offset] = scale_ue8m0

                            # Store 16 bytes (32 FP4 values = 2 x st.global.u64)
                            row_output = mOutput[row_idx, None]
                            out_base = sf_idx_in_row * (MXFP4_SF_VEC_SIZE // 2)
                            out_ptr0 = get_ptr_as_int64(row_output, out_base)
                            out_ptr1 = get_ptr_as_int64(row_output, out_base + Int32(8))
                            st_global_u64(out_ptr0, packed64_0)
                            st_global_u64(out_ptr1, packed64_1)

                        # Handle padding SF columns for this row
                        # Thread-stride loop starting from first padding column
                        if cutlass.const_expr(
                            self.num_sf_blocks_per_row != self.padded_sf_cols
                        ):
                            pad_col = num_sf_blocks_per_row + sf_idx_in_row
                            while pad_col < padded_sf_cols:
                                sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                    row_idx, pad_col, padded_sf_cols
                                )
                                mScales[sf_offset] = Uint8(0)
                                pad_col = pad_col + threads_per_row

                row_batch_idx = row_batch_idx + grid_dim_x
                # Update row_idx for next iteration
                row_idx = row_batch_idx * rows_per_block + row_in_block

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
        For linear layout: (compiled_kernel, sf_blocks_per_tb)
        For swizzled layout: (compiled_kernel, rows_per_block)
    """
    cutlass_dtype = cutlass.BFloat16 if is_bfloat16 else cutlass.Float16

    # Use symbolic M for dynamic batch sizes
    sym_m = cute.sym_int()
    sym_scale_size = cute.sym_int()

    # Common fake tensors
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K // 2), stride_order=(1, 0), assumed_align=16
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_scale_size,), assumed_align=16
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if sf_layout == SF_LAYOUT_LINEAR:
        linear_obj = MXFP4QuantizeLinearKernel(cutlass_dtype, K, enable_pdl)

        compiled_kernel = cute.compile(
            linear_obj,
            input_fake,
            output_fake,
            scales_fake,
            Int32(1),  # Dummy M
            Int32(1),  # Dummy total_sf_blocks
            Int32(1),  # Dummy num_blocks
            stream_fake,
            options="--enable-tvm-ffi",
        )

        return compiled_kernel, linear_obj.SF_BLOCKS_PER_TB
    else:
        swizzled_obj = MXFP4QuantizeSwizzledKernel(cutlass_dtype, K, enable_pdl)

        compiled_kernel = cute.compile(
            swizzled_obj,
            input_fake,
            output_fake,
            scales_fake,
            Int32(1),  # Dummy M
            Int32(128),  # Dummy padded_M
            Int32(1),  # Dummy num_blocks
            stream_fake,
            options="--enable-tvm-ffi",
        )

        return compiled_kernel, swizzled_obj.rows_per_block


@flashinfer_api
def mxfp4_quantize_cute_dsl(
    input: torch.Tensor,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to MXFP4 format using CuTe-DSL kernel.

    This is a GPU implementation with dual-path optimization:
    - LINEAR layout: flat SF-block based iteration (fast)
    - SWIZZLED layout: row-based iteration with padding fast path (optimized)

    The kernel is compiled once per (K, dtype, sf_layout, pdl) combination
    and handles varying M (batch size) at runtime without recompilation.

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

    # Get or compile kernel (device-independent)
    kernel_fn, block_unit = _get_compiled_kernel_mxfp4(
        is_bfloat16, k, sf_layout, enable_pdl
    )

    if sf_layout == SF_LAYOUT_LINEAR:
        padded_m = m
        padded_sf_cols = num_sf_blocks_per_row
        total_sf_blocks = m * num_sf_blocks_per_row
        scale_output_size = total_sf_blocks

        sf_blocks_per_tb = block_unit
        num_blocks = min(
            (total_sf_blocks + sf_blocks_per_tb - 1) // sf_blocks_per_tb,
            target_grid,
        )

        fp4_output = torch.empty(m, k // 2, dtype=torch.uint8, device=input.device)
        scale_output = torch.empty(
            scale_output_size, dtype=torch.uint8, device=input.device
        )

        kernel_fn(input, fp4_output, scale_output, m, total_sf_blocks, num_blocks)
    else:
        padded_m = ((m + ROW_TILE_SIZE - 1) // ROW_TILE_SIZE) * ROW_TILE_SIZE
        padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4
        scale_output_size = padded_m * padded_sf_cols

        rows_per_block = block_unit
        num_blocks = min(
            (padded_m + rows_per_block - 1) // rows_per_block,
            target_grid,
        )

        fp4_output = torch.empty(m, k // 2, dtype=torch.uint8, device=input.device)
        scale_output = torch.empty(
            scale_output_size, dtype=torch.uint8, device=input.device
        )

        kernel_fn(input, fp4_output, scale_output, m, padded_m, num_blocks)

    scale_output = scale_output.reshape(-1, num_sf_blocks_per_row)

    return fp4_output, scale_output


__all__ = [
    "SF_LAYOUT_128x4",
    "SF_LAYOUT_LINEAR",
    "MXFP4QuantizeLinearKernel",
    "MXFP4QuantizeSwizzledKernel",
    "mxfp4_quantize_cute_dsl",
    "_get_compiled_kernel_mxfp4",
]
