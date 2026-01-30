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

MXFP8 Quantization using CuTe-DSL
=================================

High-performance MXFP8 quantization kernel using CuTe-DSL.
Supports both linear and swizzled (128x4) scale factor layouts.

Key features:
- Half2/BFloat2 SIMD for max-abs computation
- 4-thread cooperation per scale factor block
- Dual-path optimization: linear layout (SF-block based) and swizzled layout (row-based)
- Vectorized 128-bit global loads/stores
- M-agnostic compilation: kernels are compiled once per K dimension
"""

import functools
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint8

from ..api_logging import flashinfer_api
from ..cute_dsl.fp4_common import (
    ld_global_v4_u32,
    st_global_u64,
    get_ptr_as_int64,
)
from .quantization_cute_dsl_utils import (
    # Constants
    SF_VEC_SIZE,
    INV_FLOAT8_E4M3_MAX,
    WARP_SIZE,
    ELTS_PER_THREAD,
    THREADS_PER_SF,
    SF_BLOCKS_PER_WARP,
    ROW_TILE_SIZE,
    # Low-level intrinsics
    hmax_reduce_to_f32,
    bfloat2_hmax_reduce_to_f32,
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
    reduce_max_4threads,
    compute_sf_index_swizzled_128x4_gpu,
    # High-level helpers
    half2_max_abs_4,
    bfloat2_max_abs_4,
    half2x4_to_fp8x8_packed,
    bfloat2x4_to_fp8x8_packed,
)


# Blocks per SM for occupancy target
_BLOCKS_PER_SM = 4

# Maximum threads per block (all modern NVIDIA GPUs support 1024)
_MAX_THREADS_PER_BLOCK = 1024


def _get_target_grid(device: torch.device = None) -> int:
    """
    Compute target grid size based on device SM count.

    Args:
        device: CUDA device. If None, uses current device.

    Returns:
        Target number of blocks for good occupancy (SM_count * _BLOCKS_PER_SM)
    """
    if device is None:
        device = torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return sm_count * _BLOCKS_PER_SM


# Warp configuration bounds
_MIN_WARPS = 4  # Minimum for reasonable occupancy (128 threads)
_MAX_WARPS = 32  # Maximum to avoid register pressure (1024 threads)
_DEFAULT_WARPS = 16  # Default when no optimization needed


def _compute_optimal_warps_for_k(K: int) -> int:
    """
    Compute optimal WARPS_PER_BLOCK for 100% thread utilization.

    For the swizzled kernel, we need:
        (WARPS × 8) % num_sf_blocks == 0

    where num_sf_blocks = K / 32.

    This ensures that col_units_per_block is evenly divisible by
    num_sf_blocks_per_row, so all threads are utilized.

    We prefer LARGER warp counts (up to _MAX_WARPS) for better occupancy,
    while maintaining 100% thread utilization.

    Args:
        K: Number of columns (must be divisible by 32)

    Returns:
        Optimal number of warps per block
    """
    import math

    num_sf_blocks = K // SF_VEC_SIZE  # K / 32

    # For 100% utilization: (WARPS * 8) % num_sf_blocks == 0
    # WARPS must be a multiple of: num_sf_blocks / gcd(num_sf_blocks, 8)
    gcd_val = math.gcd(num_sf_blocks, 8)
    warp_multiple = num_sf_blocks // gcd_val

    # Find LARGEST valid WARPS in range [_MIN_WARPS, _MAX_WARPS]
    # that is a multiple of warp_multiple (for best occupancy)
    if warp_multiple <= _MAX_WARPS:
        # Find largest multiple of warp_multiple that fits in [_MIN_WARPS, _MAX_WARPS]
        warps = (_MAX_WARPS // warp_multiple) * warp_multiple
        if warps >= _MIN_WARPS:
            return warps
        # If largest multiple is below _MIN_WARPS, use the smallest valid one
        warps = warp_multiple
        while warps < _MIN_WARPS:
            warps += warp_multiple
        if warps <= _MAX_WARPS:
            return warps

    # If warp_multiple is too large, fall back to default
    # This shouldn't happen for reasonable K values
    return _DEFAULT_WARPS


# =============================================================================
# CuTe-DSL Kernel Class for Linear Layout
# =============================================================================


class MXFP8QuantizeLinearKernel:
    """
    MXFP8 quantization kernel optimized for LINEAR layout.
    Uses SF-block based iteration for efficient memory access.

    This kernel is M-agnostic: compiled once per (K, dtype, pdl) combination.
    M-dependent values (total_sf_blocks) are passed at runtime.
    """

    WARPS_PER_BLOCK = 16  # 16 warps = 512 threads per block

    def __init__(
        self,
        dtype: cutlass.Numeric,
        K: int,
        enable_pdl: bool = False,
        target_grid: int = None,
    ):
        self.dtype = dtype
        self.K = K
        self.is_bfloat16 = dtype == cutlass.BFloat16
        self.enable_pdl = enable_pdl
        # Use provided target_grid or compute from current device
        self.target_grid = (
            target_grid if target_grid is not None else _get_target_grid()
        )

        assert K % SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // SF_VEC_SIZE

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        total_sf_blocks: Int32,
        stream,
    ):
        threads_per_block = self.WARPS_PER_BLOCK * WARP_SIZE
        sf_blocks_per_tb = self.WARPS_PER_BLOCK * SF_BLOCKS_PER_WARP

        # Compute grid size at runtime (target_grid is device-specific)
        num_blocks = cutlass.min(
            cute.ceil_div(total_sf_blocks, sf_blocks_per_tb), self.target_grid
        )

        self.kernel(mInput, mOutput, mScales, total_sf_blocks).launch(
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
        total_sf_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        warp_idx = tidx // WARP_SIZE
        lane_idx = tidx % WARP_SIZE

        sf_idx_in_warp = lane_idx // THREADS_PER_SF
        thread_in_sf = lane_idx % THREADS_PER_SF

        sf_blocks_per_tb = self.WARPS_PER_BLOCK * SF_BLOCKS_PER_WARP
        num_sf_blocks_per_row = self.num_sf_blocks_per_row

        sf_idx_base = (
            bidx * sf_blocks_per_tb + warp_idx * SF_BLOCKS_PER_WARP + sf_idx_in_warp
        )

        sf_idx = sf_idx_base
        while sf_idx < total_sf_blocks:
            row_idx = sf_idx // num_sf_blocks_per_row
            col_idx = sf_idx % num_sf_blocks_per_row

            base_elem = col_idx * SF_VEC_SIZE
            thread_elem_offset = thread_in_sf * ELTS_PER_THREAD
            elem_idx = base_elem + thread_elem_offset

            row_input = mInput[row_idx, None]
            input_ptr_i64 = get_ptr_as_int64(row_input, elem_idx)

            v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_i64)

            # Compute max absolute value across 8 elements (4 half2/bfloat2)
            if cutlass.const_expr(self.is_bfloat16):
                max0123 = bfloat2_max_abs_4(v0, v1, v2, v3)
                local_max = bfloat2_hmax_reduce_to_f32(max0123)
            else:
                max0123 = half2_max_abs_4(v0, v1, v2, v3)
                local_max = hmax_reduce_to_f32(max0123)

            # 4-thread reduction for this SF block
            global_max = reduce_max_4threads(local_max)

            # Compute UE8M0 scale factor
            inv_e4m3_max = Float32(INV_FLOAT8_E4M3_MAX)
            normalized_max = global_max * inv_e4m3_max
            scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
            scale_ue8m0 = scale_ue8m0_u32.to(Uint8)

            # Compute inverse scale for quantization
            inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)

            # Quantize to FP8 E4M3 and pack for vectorized store
            if cutlass.const_expr(self.is_bfloat16):
                fp8_packed = bfloat2x4_to_fp8x8_packed(v0, v1, v2, v3, inv_scale)
            else:
                fp8_packed = half2x4_to_fp8x8_packed(v0, v1, v2, v3, inv_scale)

            row_output = mOutput[row_idx, None]
            output_ptr_i64 = get_ptr_as_int64(row_output, elem_idx)
            st_global_u64(output_ptr_i64, fp8_packed)

            if thread_in_sf == Int32(0):
                mScales[sf_idx] = scale_ue8m0

            sf_idx = sf_idx + grid_dim_x * sf_blocks_per_tb

        # PDL: Signal that dependent kernels can start early
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# CuTe-DSL Kernel Class for Swizzled Layout
# =============================================================================


class MXFP8QuantizeSwizzledKernel:
    """
    MXFP8 quantization kernel optimized for SWIZZLED layout.

    Key optimizations:
    - Multi-row processing: threads process multiple rows per block when K is small
    - Row-based iteration with grid-stride loop
    - Padding row fast path - only zero out scale factors

    Thread utilization optimization:
    - Dynamic WARPS_PER_BLOCK based on K for 100% thread utilization
    - For small K: Multiple rows processed per block iteration
    - For large K: Single row with column loop

    This kernel is M-agnostic: compiled once per (K, dtype, pdl) combination.
    M-dependent values (M, padded_M) are passed at runtime.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        K: int,
        enable_pdl: bool = False,
        target_grid: int = None,
    ):
        self.dtype = dtype
        self.K = K
        self.is_bfloat16 = dtype == cutlass.BFloat16
        self.enable_pdl = enable_pdl
        # Use provided target_grid or compute from current device
        self.target_grid = (
            target_grid if target_grid is not None else _get_target_grid()
        )

        assert K % SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // SF_VEC_SIZE
        self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4

        # Compute optimal warps for 100% thread utilization
        self.warps_per_block = _compute_optimal_warps_for_k(K)

        # Multi-row processing constants (compile-time)
        threads_per_block = self.warps_per_block * WARP_SIZE
        col_units_per_block = threads_per_block // THREADS_PER_SF

        # threads_per_row = num_sf_blocks_per_row * THREADS_PER_SF = K / 8
        self.threads_per_row = self.num_sf_blocks_per_row * THREADS_PER_SF

        # rows_per_block = col_units_per_block // num_sf_blocks_per_row
        # With optimal warps, this should divide evenly for small K
        if self.num_sf_blocks_per_row <= col_units_per_block:
            self.rows_per_block = col_units_per_block // self.num_sf_blocks_per_row
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
        stream,
    ):
        threads_per_block = self.warps_per_block * WARP_SIZE
        rows_per_block = self.rows_per_block

        # Compute grid size at runtime (target_grid is device-specific)
        # Grid covers row batches, not individual rows
        total_row_batches = cute.ceil_div(padded_M, rows_per_block)
        num_blocks = cutlass.min(total_row_batches, self.target_grid)

        self.kernel(mInput, mOutput, mScales, M, padded_M).launch(
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
        padded_M: Int32,
    ):
        """
        Multi-row kernel for swizzled layout.

        When K is small: each block processes multiple rows simultaneously.
        When K is large: each block processes one row with column loop.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        # Compile-time constants
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        padded_sf_cols = self.padded_sf_cols
        threads_per_row = self.threads_per_row
        rows_per_block = self.rows_per_block

        if cutlass.const_expr(self.needs_col_loop):
            # Large K path: single row per block iteration with column loop
            # This is the original algorithm for K > 4096
            col_unit_idx = tidx // THREADS_PER_SF
            thread_in_unit = tidx % THREADS_PER_SF
            threads_per_block = self.warps_per_block * WARP_SIZE
            col_units_per_block = threads_per_block // THREADS_PER_SF

            row_idx = bidx
            while row_idx < padded_M:
                is_padding_row = row_idx >= M

                if is_padding_row:
                    # Fast path: padding row - only zero out scale factors
                    sf_col_idx = col_unit_idx
                    while sf_col_idx < padded_sf_cols:
                        if thread_in_unit == Int32(0):
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, sf_col_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)
                        sf_col_idx = sf_col_idx + col_units_per_block
                else:
                    # Normal path: process actual data row with column loop
                    sf_col_idx = col_unit_idx
                    while sf_col_idx < num_sf_blocks_per_row:
                        elem_idx = (
                            sf_col_idx * SF_VEC_SIZE + thread_in_unit * ELTS_PER_THREAD
                        )

                        row_input = mInput[row_idx, None]
                        input_ptr_i64 = get_ptr_as_int64(row_input, elem_idx)
                        v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_i64)

                        if cutlass.const_expr(self.is_bfloat16):
                            max0123 = bfloat2_max_abs_4(v0, v1, v2, v3)
                            local_max = bfloat2_hmax_reduce_to_f32(max0123)
                        else:
                            max0123 = half2_max_abs_4(v0, v1, v2, v3)
                            local_max = hmax_reduce_to_f32(max0123)

                        global_max = reduce_max_4threads(local_max)
                        inv_e4m3_max = Float32(INV_FLOAT8_E4M3_MAX)
                        normalized_max = global_max * inv_e4m3_max
                        scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
                        scale_ue8m0 = scale_ue8m0_u32.to(Uint8)
                        inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)

                        if cutlass.const_expr(self.is_bfloat16):
                            fp8_packed = bfloat2x4_to_fp8x8_packed(
                                v0, v1, v2, v3, inv_scale
                            )
                        else:
                            fp8_packed = half2x4_to_fp8x8_packed(
                                v0, v1, v2, v3, inv_scale
                            )

                        row_output = mOutput[row_idx, None]
                        output_ptr_i64 = get_ptr_as_int64(row_output, elem_idx)
                        st_global_u64(output_ptr_i64, fp8_packed)

                        if thread_in_unit == Int32(0):
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, sf_col_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = scale_ue8m0

                        sf_col_idx = sf_col_idx + col_units_per_block

                    # Handle padding columns
                    sf_col_idx = num_sf_blocks_per_row + col_unit_idx
                    while sf_col_idx < padded_sf_cols:
                        if thread_in_unit == Int32(0):
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, sf_col_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)
                        sf_col_idx = sf_col_idx + col_units_per_block

                row_idx = row_idx + grid_dim_x
        else:
            # Small K path: multi-row processing (K <= 4096)
            # Each block processes rows_per_block rows simultaneously
            # Thread mapping: tidx -> (row_in_block, sf_col_idx, thread_in_unit)
            row_in_block = tidx // threads_per_row
            local_tidx = tidx % threads_per_row
            sf_col_idx = local_tidx // THREADS_PER_SF
            thread_in_unit = local_tidx % THREADS_PER_SF

            # Grid-stride loop over row batches
            row_batch_idx = bidx
            # Initialize row_idx before while loop (CuTe DSL requires variables
            # modified in while loops to be defined before the loop)
            row_idx = row_batch_idx * rows_per_block + row_in_block
            while row_batch_idx * rows_per_block < padded_M:
                # Check if this thread's row is valid
                if row_idx < padded_M:
                    is_padding_row = row_idx >= M

                    if is_padding_row:
                        # Fast path: padding row - zero out scale factors
                        # Each thread handles one SF column (no column loop needed)
                        if sf_col_idx < padded_sf_cols and thread_in_unit == Int32(0):
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, sf_col_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)
                    else:
                        # Normal path: process actual data
                        if sf_col_idx < num_sf_blocks_per_row:
                            elem_idx = (
                                sf_col_idx * SF_VEC_SIZE
                                + thread_in_unit * ELTS_PER_THREAD
                            )

                            row_input = mInput[row_idx, None]
                            input_ptr_i64 = get_ptr_as_int64(row_input, elem_idx)
                            v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_i64)

                            if cutlass.const_expr(self.is_bfloat16):
                                max0123 = bfloat2_max_abs_4(v0, v1, v2, v3)
                                local_max = bfloat2_hmax_reduce_to_f32(max0123)
                            else:
                                max0123 = half2_max_abs_4(v0, v1, v2, v3)
                                local_max = hmax_reduce_to_f32(max0123)

                            global_max = reduce_max_4threads(local_max)
                            inv_e4m3_max = Float32(INV_FLOAT8_E4M3_MAX)
                            normalized_max = global_max * inv_e4m3_max
                            scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
                            scale_ue8m0 = scale_ue8m0_u32.to(Uint8)
                            inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)

                            if cutlass.const_expr(self.is_bfloat16):
                                fp8_packed = bfloat2x4_to_fp8x8_packed(
                                    v0, v1, v2, v3, inv_scale
                                )
                            else:
                                fp8_packed = half2x4_to_fp8x8_packed(
                                    v0, v1, v2, v3, inv_scale
                                )

                            row_output = mOutput[row_idx, None]
                            output_ptr_i64 = get_ptr_as_int64(row_output, elem_idx)
                            st_global_u64(output_ptr_i64, fp8_packed)

                            if thread_in_unit == Int32(0):
                                sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                    row_idx, sf_col_idx, padded_sf_cols
                                )
                                mScales[sf_offset] = scale_ue8m0

                        # Handle padding SF columns (for this row)
                        # Threads with sf_col_idx in [num_sf_blocks_per_row, padded_sf_cols)
                        if (
                            sf_col_idx >= num_sf_blocks_per_row
                            and sf_col_idx < padded_sf_cols
                            and thread_in_unit == Int32(0)
                        ):
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, sf_col_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)

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
def _get_compiled_kernel_linear(
    is_bfloat16: bool,
    K: int,
    enable_pdl: bool = False,
    target_grid: int = None,
) -> Callable:
    """
    Get or compile LINEAR layout kernel with TVM-FFI.

    Cached by (K, dtype, pdl, target_grid) - M-agnostic compilation.
    """
    cutlass_dtype = cutlass.BFloat16 if is_bfloat16 else cutlass.Float16
    kernel_obj = MXFP8QuantizeLinearKernel(
        cutlass_dtype, K, enable_pdl, target_grid=target_grid
    )

    # Use symbolic M for dynamic batch sizes
    sym_m = cute.sym_int()

    # Create fake tensors for compilation
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    # Scale output size is dynamic (M-dependent)
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
        Int32(1),  # Dummy total_sf_blocks - actual value passed at runtime
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


@functools.cache
def _get_compiled_kernel_swizzled(
    is_bfloat16: bool,
    K: int,
    enable_pdl: bool = False,
    target_grid: int = None,
) -> Callable:
    """
    Get or compile SWIZZLED layout kernel with TVM-FFI.

    Cached by (K, dtype, pdl, target_grid) - M-agnostic compilation.
    """
    cutlass_dtype = cutlass.BFloat16 if is_bfloat16 else cutlass.Float16
    kernel_obj = MXFP8QuantizeSwizzledKernel(
        cutlass_dtype, K, enable_pdl, target_grid=target_grid
    )

    # Use symbolic M for dynamic batch sizes
    sym_m = cute.sym_int()

    # Create fake tensors for compilation
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    # Scale output size is dynamic (M-dependent)
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
        Int32(1),  # Dummy M - actual value passed at runtime
        Int32(128),  # Dummy padded_M - actual value passed at runtime
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


@flashinfer_api
def mxfp8_quantize_cute_dsl(
    input: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    enable_pdl: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to MXFP8 format using CuTe-DSL kernel.

    This is a GPU implementation with dual-path optimization:
    - LINEAR layout: SF-block based iteration (fast)
    - SWIZZLED layout: Row-based iteration with padding fast path (optimized)

    The kernel is compiled once per (K, dtype, pdl) combination and handles
    varying M (batch size) at runtime without recompilation.

    Args:
        input: Input tensor of shape [M, K] with dtype fp16/bf16
        is_sf_swizzled_layout: Whether to use 128x4 swizzled layout (True) or linear (False)
        alignment: Alignment for K dimension (default 32, must be multiple of SF_VEC_SIZE)
        enable_pdl: Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability (SM >= 9.0).

    Returns:
        Tuple of:
            - fp8_tensor: Quantized tensor of shape [M, padded_K] with dtype float8_e4m3fn
            - scale_tensor: Scale factors as uint8 tensor
    """
    from ..utils import device_support_pdl

    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"Input dtype must be float16 or bfloat16, got {input.dtype}"
    )
    assert input.is_cuda, "Input must be on CUDA device"
    assert alignment % SF_VEC_SIZE == 0, (
        f"alignment must be divisible by SF_VEC_SIZE={SF_VEC_SIZE}"
    )

    # Auto-detect PDL support based on device capability
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)

    if input.dim() > 2:
        m = input.numel() // input.shape[-1]
        k = input.shape[-1]
        input = input.reshape(m, k)
    else:
        m, k = input.shape

    assert k % SF_VEC_SIZE == 0, (
        f"K ({k}) must be divisible by SF_VEC_SIZE={SF_VEC_SIZE}"
    )

    padded_k = ((k + alignment - 1) // alignment) * alignment

    if padded_k > k:
        # Pad input with zeros - padding columns must be zero to produce zero FP8 output
        input_padded = torch.zeros(m, padded_k, dtype=input.dtype, device=input.device)
        input_padded[:, :k] = input
    else:
        input_padded = input.contiguous()

    is_bfloat16 = input.dtype == torch.bfloat16

    # Compute device-specific target grid for kernel compilation
    target_grid = _get_target_grid(input.device)

    # Compute M-dependent values outside the cached kernel
    num_sf_blocks_per_row = padded_k // SF_VEC_SIZE

    if is_sf_swizzled_layout:
        # Swizzled layout: compute padded_M and scale_output_size
        padded_m = ((m + ROW_TILE_SIZE - 1) // ROW_TILE_SIZE) * ROW_TILE_SIZE
        padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4
        scale_output_size = padded_m * padded_sf_cols

        kernel_fn = _get_compiled_kernel_swizzled(
            is_bfloat16, padded_k, enable_pdl, target_grid
        )

        fp8_output = torch.empty(m, padded_k, dtype=torch.uint8, device=input.device)
        scale_output = torch.empty(
            scale_output_size, dtype=torch.uint8, device=input.device
        )

        kernel_fn(input_padded, fp8_output, scale_output, m, padded_m)
    else:
        # Linear layout: compute total_sf_blocks
        total_sf_blocks = m * num_sf_blocks_per_row
        scale_output_size = total_sf_blocks

        kernel_fn = _get_compiled_kernel_linear(
            is_bfloat16, padded_k, enable_pdl, target_grid
        )

        fp8_output = torch.empty(m, padded_k, dtype=torch.uint8, device=input.device)
        scale_output = torch.empty(
            scale_output_size, dtype=torch.uint8, device=input.device
        )

        kernel_fn(input_padded, fp8_output, scale_output, total_sf_blocks)

    fp8_tensor = fp8_output.view(torch.float8_e4m3fn)

    return fp8_tensor, scale_output


__all__ = [
    "MXFP8QuantizeLinearKernel",
    "MXFP8QuantizeSwizzledKernel",
    "mxfp8_quantize_cute_dsl",
    "_get_compiled_kernel_linear",
    "_get_compiled_kernel_swizzled",
]
