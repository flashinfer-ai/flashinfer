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
Supports linear, swizzled 128x4, and swizzled 8x4 scale factor layouts.

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

from ...api_logging import flashinfer_api
from ...cute_dsl.fp4_common import (
    fmax_f32,
    ld_global_v4_u32,
    st_global_u64,
    get_ptr_as_int64,
)
from ...cute_dsl.utils import get_num_sm
from ..quantization_cute_dsl_utils import (
    # Constants
    SF_VEC_SIZE,
    INV_FLOAT8_E4M3_MAX,
    WARP_SIZE,
    ELTS_PER_THREAD,
    THREADS_PER_SF,
    SF_BLOCKS_PER_WARP,
    ELTS_PER_THREAD_SMALL,
    THREADS_PER_SF_SMALL,
    SF_BLOCKS_PER_WARP_SMALL,
    MXFP8_2T_SF_THRESHOLD,
    ROW_TILE_SIZE,
    # Low-level intrinsics
    hmax_reduce_to_f32,
    bfloat2_hmax_reduce_to_f32,
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
    reduce_max_2threads,
    reduce_max_4threads,
    compute_sf_index_swizzled_128x4_gpu,
    compute_sf_index_swizzled_8x4_gpu,
    # High-level helpers
    half2_max_abs_4,
    half2_max_abs_8,
    bfloat2_max_abs_4,
    bfloat2_max_abs_8,
    half2x4_to_fp8x8_packed,
    bfloat2x4_to_fp8x8_packed,
    float_max_abs_8,
    floatx8_to_fp8x8_packed,
    ld_global_v8_u32,
)


# Blocks per SM for occupancy target
_BLOCKS_PER_SM = 4

# Maximum threads per block (all modern NVIDIA GPUs support 1024)
_MAX_THREADS_PER_BLOCK = 1024


# Warp configuration bounds
_MIN_WARPS = 4  # Minimum for reasonable occupancy (128 threads)
_MAX_WARPS = 32  # Maximum to avoid register pressure (1024 threads)
_DEFAULT_WARPS = 16  # Default when no optimization needed

# Scale factor layout codes (match nvfp4/mxfp4 conventions)
SF_LAYOUT_128x4 = 0
SF_LAYOUT_8x4 = 1
SF_LAYOUT_LINEAR = 2


# SM107 128x4 tile kernel (see MXFP8QuantizeTile128x4Kernel): 4-column groups
# per tile, threads per CTA, and persistent CTAs per SM.
_TILE128X4_COL_GROUPS = 1
_TILE128X4_THREADS = 256
_TILE128X4_BLOCKS_PER_SM = 4


def _use_sm107_tile128x4(m: int, k: int, elt_bytes: int) -> bool:
    """Route SM107 128x4 inputs to the tile kernel.

    Cold-L2 sweeps across input dtypes: the tile kernel wins (1.04-1.38x) once
    a row is at least 8 KiB and the input at least 16 MiB, or a 4 KiB row and a
    64 MiB input; below that the row kernel's full-row streaming has better
    DRAM locality than 256 B row segments (e.g. FP8 4096x4096 or BF16
    8192x2048 lose ~5%). K must also span at least 64 SF columns of tiles
    (K >= 2048 elements): FP32 16384x1024 meets the byte thresholds but loses.
    """
    if m < 1024 or k < 2048:
        return False
    row_bytes = k * elt_bytes
    total = m * row_bytes
    if row_bytes >= 8 << 10:
        return total >= 16 << 20
    return row_bytes >= 4 << 10 and total >= 64 << 20


def _use_sm107_small_blocks(m: int, k: int, is_float32: bool = False) -> bool:
    # 256-thread CTAs with a full column loop beat the default (up to 1024
    # threads, one row per CTA) once rows are wide. Cold-L2 sweeps on Rubin:
    # K > 16384 gains 1.2-1.5x for every input dtype, including power-of-two
    # widths; 8192 < K <= 12288 gains 1.05-1.15x for FP16/BF16 only (FP32 is
    # neutral there); K = 14336/16384 and K <= 8192 favor the default. Short
    # batches (M < 1536) keep the default.
    if m < 1536:
        return False
    if k > 16384:
        return True
    return not is_float32 and 8192 < k <= 12288


def _compute_optimal_warps(K: int, sf_blocks_per_warp: int = SF_BLOCKS_PER_WARP) -> int:
    """
    Compute optimal WARPS_PER_BLOCK for 100% thread utilization.

    For the swizzled kernel, we need:
        (WARPS x sf_blocks_per_warp) % num_sf_blocks == 0

    where num_sf_blocks = K / 32.

    This ensures that col_units_per_block is evenly divisible by
    num_sf_blocks_per_row, so all threads are utilized.

    We prefer LARGER warp counts (up to _MAX_WARPS) for better occupancy,
    while maintaining 100% thread utilization.

    Args:
        K: Number of columns (must be divisible by 32)
        sf_blocks_per_warp: SF blocks per warp for the selected thread
            configuration (16 for 2T/SF, 8 for 4T/SF)

    Returns:
        Optimal number of warps per block
    """
    import math

    num_sf_blocks = K // SF_VEC_SIZE  # K / 32

    # For 100% utilization: (WARPS * sf_blocks_per_warp) % num_sf_blocks == 0
    # WARPS must be a multiple of: num_sf_blocks / gcd(num_sf_blocks, sf_blocks_per_warp)
    gcd_val = math.gcd(num_sf_blocks, sf_blocks_per_warp)
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
# CuTe-DSL Kernel Class for Linear Layout — Flat SF-Block Iteration
# =============================================================================


class MXFP8QuantizeLinearKernel:
    """
    MXFP8 quantization kernel optimized for LINEAR layout.

    Uses flat SF-block iteration for efficient memory access. Row and
    column indices are derived from the flat SF index via integer division.

    No padding passes are needed since for linear layout:
    - padded_m == m (no row padding)
    - padded_sf_cols == num_sf_blocks_per_row (no column padding)

    Adaptive thread configuration (compile-time selected via use_2t_per_sf):
    - 2T/SF (large problems): 2 threads per SF block, 16 elements per thread,
      1 shuffle reduction, 16 SF blocks per warp
    - 4T/SF (small problems): 4 threads per SF block, 8 elements per thread,
      2 shuffle reductions, 8 SF blocks per warp

    This kernel is M-agnostic: compiled once per (K, dtype, pdl, use_2t)
    combination.
    """

    WARPS_PER_BLOCK = 16  # 16 warps = 512 threads per block

    def __init__(
        self,
        dtype: cutlass.Numeric,
        K: int,
        enable_pdl: bool = False,
        use_2t_per_sf: bool = True,
    ):
        self.is_bfloat16 = dtype == cutlass.BFloat16
        self.is_float32 = dtype == cutlass.Float32
        self.enable_pdl = enable_pdl
        self.use_2t_per_sf = use_2t_per_sf

        if use_2t_per_sf:
            self._elts_per_thread = ELTS_PER_THREAD
            self._threads_per_sf = THREADS_PER_SF
            self._sf_blocks_per_warp = SF_BLOCKS_PER_WARP
        else:
            self._elts_per_thread = ELTS_PER_THREAD_SMALL
            self._threads_per_sf = THREADS_PER_SF_SMALL
            self._sf_blocks_per_warp = SF_BLOCKS_PER_WARP_SMALL

        self.SF_BLOCKS_PER_TB = self.WARPS_PER_BLOCK * self._sf_blocks_per_warp

        assert K % SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // SF_VEC_SIZE

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        total_sf_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        threads_per_block = self.WARPS_PER_BLOCK * WARP_SIZE

        self.kernel(mInput, mOutput, mScales, total_sf_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[threads_per_block, 1, 1],
            max_number_threads=[_MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=_BLOCKS_PER_SM,
            smem=0,
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

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        warp_idx = tidx // WARP_SIZE
        lane_idx = tidx % WARP_SIZE

        threads_per_sf = self._threads_per_sf
        sf_blocks_per_warp = self._sf_blocks_per_warp
        elts_per_thread = self._elts_per_thread

        sf_idx_in_warp = lane_idx // threads_per_sf
        thread_in_sf = lane_idx % threads_per_sf

        sf_blocks_per_tb = self.WARPS_PER_BLOCK * sf_blocks_per_warp
        num_sf_blocks_per_row = self.num_sf_blocks_per_row

        sf_idx_base = (
            bidx * sf_blocks_per_tb + warp_idx * sf_blocks_per_warp + sf_idx_in_warp
        )

        sf_idx = sf_idx_base
        while sf_idx < total_sf_blocks:
            row_idx = sf_idx // num_sf_blocks_per_row
            col_idx = sf_idx % num_sf_blocks_per_row

            base_elem = col_idx * SF_VEC_SIZE
            thread_elem_offset = thread_in_sf * elts_per_thread
            elem_idx = base_elem + thread_elem_offset

            row_input = mInput[row_idx, None]

            if cutlass.const_expr(self.use_2t_per_sf):
                # 2T/SF path: 16 elements/thread, 1-shuffle reduction.
                # fp16/bf16: 2x128-bit loads (8 elts each); fp32: 4x128-bit
                # loads (4 elts each), so lanes v0..v15 hold 16 floats.
                if cutlass.const_expr(self.is_float32):
                    v0, v1, v2, v3, v4, v5, v6, v7 = ld_global_v8_u32(
                        get_ptr_as_int64(row_input, elem_idx)
                    )
                    v8, v9, v10, v11, v12, v13, v14, v15 = ld_global_v8_u32(
                        get_ptr_as_int64(row_input, elem_idx + Int32(8))
                    )
                    local_max = fmax_f32(
                        float_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7),
                        float_max_abs_8(v8, v9, v10, v11, v12, v13, v14, v15),
                    )
                else:
                    input_ptr_lo = get_ptr_as_int64(row_input, elem_idx)
                    input_ptr_hi = get_ptr_as_int64(row_input, elem_idx + Int32(8))
                    v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_lo)
                    v4, v5, v6, v7 = ld_global_v4_u32(input_ptr_hi)

                    if cutlass.const_expr(self.is_bfloat16):
                        max_all = bfloat2_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7)
                        local_max = bfloat2_hmax_reduce_to_f32(max_all)
                    else:
                        max_all = half2_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7)
                        local_max = hmax_reduce_to_f32(max_all)

                global_max = reduce_max_2threads(local_max)
            else:
                # 4T/SF path: 8 elements/thread, 2-shuffle reduction.
                # fp16/bf16: 1x128-bit load; fp32: 2x128-bit loads.
                if cutlass.const_expr(self.is_float32):
                    v0, v1, v2, v3, v4, v5, v6, v7 = ld_global_v8_u32(
                        get_ptr_as_int64(row_input, elem_idx)
                    )
                    local_max = float_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7)
                else:
                    input_ptr_i64 = get_ptr_as_int64(row_input, elem_idx)
                    v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_i64)

                    if cutlass.const_expr(self.is_bfloat16):
                        max0123 = bfloat2_max_abs_4(v0, v1, v2, v3)
                        local_max = bfloat2_hmax_reduce_to_f32(max0123)
                    else:
                        max0123 = half2_max_abs_4(v0, v1, v2, v3)
                        local_max = hmax_reduce_to_f32(max0123)

                global_max = reduce_max_4threads(local_max)

            # Compute UE8M0 scale factor
            inv_e4m3_max = Float32(INV_FLOAT8_E4M3_MAX)
            normalized_max = global_max * inv_e4m3_max
            scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
            scale_ue8m0 = scale_ue8m0_u32.to(Uint8)

            # Compute inverse scale for quantization
            inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)

            # Quantize to FP8 E4M3 and pack for vectorized store
            if cutlass.const_expr(self.use_2t_per_sf):
                if cutlass.const_expr(self.is_float32):
                    fp8_lo = floatx8_to_fp8x8_packed(
                        v0, v1, v2, v3, v4, v5, v6, v7, inv_scale
                    )
                    fp8_hi = floatx8_to_fp8x8_packed(
                        v8, v9, v10, v11, v12, v13, v14, v15, inv_scale
                    )
                elif cutlass.const_expr(self.is_bfloat16):
                    fp8_lo = bfloat2x4_to_fp8x8_packed(v0, v1, v2, v3, inv_scale)
                    fp8_hi = bfloat2x4_to_fp8x8_packed(v4, v5, v6, v7, inv_scale)
                else:
                    fp8_lo = half2x4_to_fp8x8_packed(v0, v1, v2, v3, inv_scale)
                    fp8_hi = half2x4_to_fp8x8_packed(v4, v5, v6, v7, inv_scale)

                row_output = mOutput[row_idx, None]
                st_global_u64(get_ptr_as_int64(row_output, elem_idx), fp8_lo)
                st_global_u64(get_ptr_as_int64(row_output, elem_idx + Int32(8)), fp8_hi)
            else:
                if cutlass.const_expr(self.is_float32):
                    fp8_packed = floatx8_to_fp8x8_packed(
                        v0, v1, v2, v3, v4, v5, v6, v7, inv_scale
                    )
                elif cutlass.const_expr(self.is_bfloat16):
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
# CuTe-DSL Kernel Class for Swizzled Layout — Row-Based Iteration
# =============================================================================


class MXFP8QuantizeSwizzledKernel:
    """
    MXFP8 quantization kernel optimized for SWIZZLED (128x4 or 8x4) layout.

    Key optimizations:
    - Multi-row processing: threads process multiple rows per block when K is small
    - Row-based iteration with grid-stride loop
    - Padding row fast path - only zero out scale factors

    Thread utilization optimization:
    - Dynamic WARPS_PER_BLOCK based on K for 100% thread utilization
    - For small K: Multiple rows processed per block iteration
    - For large K: Single row with column loop

    For MXFP8, each SF block (32 elements) is processed by _threads_per_sf
    threads (2 or 4), so threads_per_row = num_sf_blocks_per_row * _threads_per_sf.

    This kernel is M-agnostic: compiled once per (K, dtype, pdl, use_2t, small_blocks)
    combination. M-dependent values (M, padded_M) are passed at runtime.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        K: int,
        enable_pdl: bool = False,
        use_2t_per_sf: bool = True,
        sf_layout: int = SF_LAYOUT_128x4,
        small_blocks: bool = False,
    ):
        self.is_bfloat16 = dtype == cutlass.BFloat16
        self.is_float32 = dtype == cutlass.Float32
        self.enable_pdl = enable_pdl
        self.use_2t_per_sf = use_2t_per_sf
        self.sf_layout = sf_layout
        self.sf_is_128x4 = sf_layout == SF_LAYOUT_128x4
        self.sf_is_8x4 = sf_layout == SF_LAYOUT_8x4
        self.small_blocks = small_blocks

        if use_2t_per_sf:
            self._elts_per_thread = ELTS_PER_THREAD
            self._threads_per_sf = THREADS_PER_SF
            self._sf_blocks_per_warp = SF_BLOCKS_PER_WARP
        else:
            self._elts_per_thread = ELTS_PER_THREAD_SMALL
            self._threads_per_sf = THREADS_PER_SF_SMALL
            self._sf_blocks_per_warp = SF_BLOCKS_PER_WARP_SMALL

        assert K % SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // SF_VEC_SIZE
        self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4

        # Compute optimal warps for 100% thread utilization
        self.warps_per_block = (
            8 if small_blocks else _compute_optimal_warps(K, self._sf_blocks_per_warp)
        )

        # Multi-row processing constants (compile-time)
        threads_per_block = self.warps_per_block * WARP_SIZE
        col_units_per_block = threads_per_block // self._threads_per_sf

        # threads_per_row = num_sf_blocks_per_row * _threads_per_sf
        self.threads_per_row = self.num_sf_blocks_per_row * self._threads_per_sf

        # rows_per_block = col_units_per_block // num_sf_blocks_per_row
        # With optimal warps, this should divide evenly for small K
        if self.num_sf_blocks_per_row <= col_units_per_block:
            self.rows_per_block = col_units_per_block // self.num_sf_blocks_per_row
            self.needs_col_loop = False
        else:
            self.rows_per_block = 1
            self.needs_col_loop = True

    @cute.jit
    def _compute_sf_offset(
        self, row_idx: Int32, col_idx: Int32, padded_cols: Int32
    ) -> Int32:
        """Compute scale factor offset based on layout (compile-time dispatch)."""
        if cutlass.const_expr(self.sf_is_128x4):
            return compute_sf_index_swizzled_128x4_gpu(row_idx, col_idx, padded_cols)
        else:
            return compute_sf_index_swizzled_8x4_gpu(row_idx, col_idx, padded_cols)

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
        threads_per_block = self.warps_per_block * WARP_SIZE

        self.kernel(mInput, mOutput, mScales, M, padded_M).launch(
            grid=[num_blocks, 1, 1],
            block=[threads_per_block, 1, 1],
            max_number_threads=[
                256 if self.small_blocks else _MAX_THREADS_PER_BLOCK,
                1,
                1,
            ],
            min_blocks_per_mp=1 if self.small_blocks else _BLOCKS_PER_SM,
            smem=0,
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

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        # Compile-time constants
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        padded_sf_cols = self.padded_sf_cols
        threads_per_row = self.threads_per_row
        rows_per_block = self.rows_per_block

        _threads_per_sf = self._threads_per_sf
        _elts_per_thread = self._elts_per_thread

        if cutlass.const_expr(self.needs_col_loop):
            # Large K path: single row per block iteration with column loop
            col_unit_idx = tidx // _threads_per_sf
            thread_in_unit = tidx % _threads_per_sf
            threads_per_block = self.warps_per_block * WARP_SIZE
            col_units_per_block = threads_per_block // _threads_per_sf

            row_idx = bidx
            while row_idx < padded_M:
                is_padding_row = row_idx >= M

                if is_padding_row:
                    # Fast path: padding row - only zero out scale factors
                    sf_col_idx = col_unit_idx
                    while sf_col_idx < padded_sf_cols:
                        if thread_in_unit == Int32(0):
                            sf_offset = self._compute_sf_offset(
                                row_idx, sf_col_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)
                        sf_col_idx = sf_col_idx + col_units_per_block
                else:
                    # Normal path: process actual data row with column loop
                    sf_col_idx = col_unit_idx
                    while sf_col_idx < num_sf_blocks_per_row:
                        elem_idx = (
                            sf_col_idx * SF_VEC_SIZE + thread_in_unit * _elts_per_thread
                        )

                        row_input = mInput[row_idx, None]

                        if cutlass.const_expr(self.use_2t_per_sf):
                            if cutlass.const_expr(self.is_float32):
                                # fp32: 16 elements = 2x256-bit loads
                                v0, v1, v2, v3, v4, v5, v6, v7 = ld_global_v8_u32(
                                    get_ptr_as_int64(row_input, elem_idx)
                                )
                                (
                                    v8,
                                    v9,
                                    v10,
                                    v11,
                                    v12,
                                    v13,
                                    v14,
                                    v15,
                                ) = ld_global_v8_u32(
                                    get_ptr_as_int64(row_input, elem_idx + Int32(8))
                                )
                                local_max = fmax_f32(
                                    float_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7),
                                    float_max_abs_8(
                                        v8, v9, v10, v11, v12, v13, v14, v15
                                    ),
                                )
                            else:
                                input_ptr_lo = get_ptr_as_int64(row_input, elem_idx)
                                input_ptr_hi = get_ptr_as_int64(
                                    row_input, elem_idx + Int32(8)
                                )
                                v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_lo)
                                v4, v5, v6, v7 = ld_global_v4_u32(input_ptr_hi)
                                if cutlass.const_expr(self.is_bfloat16):
                                    local_max = bfloat2_hmax_reduce_to_f32(
                                        bfloat2_max_abs_8(
                                            v0, v1, v2, v3, v4, v5, v6, v7
                                        )
                                    )
                                else:
                                    local_max = hmax_reduce_to_f32(
                                        half2_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7)
                                    )
                            global_max = reduce_max_2threads(local_max)
                        else:
                            if cutlass.const_expr(self.is_float32):
                                # fp32: 8 elements = 1x256-bit load
                                v0, v1, v2, v3, v4, v5, v6, v7 = ld_global_v8_u32(
                                    get_ptr_as_int64(row_input, elem_idx)
                                )
                                local_max = float_max_abs_8(
                                    v0, v1, v2, v3, v4, v5, v6, v7
                                )
                            else:
                                input_ptr_i64 = get_ptr_as_int64(row_input, elem_idx)
                                v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_i64)
                                if cutlass.const_expr(self.is_bfloat16):
                                    local_max = bfloat2_hmax_reduce_to_f32(
                                        bfloat2_max_abs_4(v0, v1, v2, v3)
                                    )
                                else:
                                    local_max = hmax_reduce_to_f32(
                                        half2_max_abs_4(v0, v1, v2, v3)
                                    )
                            global_max = reduce_max_4threads(local_max)

                        inv_e4m3_max = Float32(INV_FLOAT8_E4M3_MAX)
                        normalized_max = global_max * inv_e4m3_max
                        scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
                        scale_ue8m0 = scale_ue8m0_u32.to(Uint8)
                        inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)

                        row_output = mOutput[row_idx, None]
                        if cutlass.const_expr(self.use_2t_per_sf):
                            if cutlass.const_expr(self.is_float32):
                                fp8_lo = floatx8_to_fp8x8_packed(
                                    v0, v1, v2, v3, v4, v5, v6, v7, inv_scale
                                )
                                fp8_hi = floatx8_to_fp8x8_packed(
                                    v8, v9, v10, v11, v12, v13, v14, v15, inv_scale
                                )
                            elif cutlass.const_expr(self.is_bfloat16):
                                fp8_lo = bfloat2x4_to_fp8x8_packed(
                                    v0, v1, v2, v3, inv_scale
                                )
                                fp8_hi = bfloat2x4_to_fp8x8_packed(
                                    v4, v5, v6, v7, inv_scale
                                )
                            else:
                                fp8_lo = half2x4_to_fp8x8_packed(
                                    v0, v1, v2, v3, inv_scale
                                )
                                fp8_hi = half2x4_to_fp8x8_packed(
                                    v4, v5, v6, v7, inv_scale
                                )
                            st_global_u64(
                                get_ptr_as_int64(row_output, elem_idx), fp8_lo
                            )
                            st_global_u64(
                                get_ptr_as_int64(row_output, elem_idx + Int32(8)),
                                fp8_hi,
                            )
                        else:
                            if cutlass.const_expr(self.is_float32):
                                fp8_packed = floatx8_to_fp8x8_packed(
                                    v0, v1, v2, v3, v4, v5, v6, v7, inv_scale
                                )
                            elif cutlass.const_expr(self.is_bfloat16):
                                fp8_packed = bfloat2x4_to_fp8x8_packed(
                                    v0, v1, v2, v3, inv_scale
                                )
                            else:
                                fp8_packed = half2x4_to_fp8x8_packed(
                                    v0, v1, v2, v3, inv_scale
                                )
                            st_global_u64(
                                get_ptr_as_int64(row_output, elem_idx), fp8_packed
                            )

                        if thread_in_unit == Int32(0):
                            sf_offset = self._compute_sf_offset(
                                row_idx, sf_col_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = scale_ue8m0

                        sf_col_idx = sf_col_idx + col_units_per_block

                    # Handle padding columns
                    sf_col_idx = num_sf_blocks_per_row + col_unit_idx
                    while sf_col_idx < padded_sf_cols:
                        if thread_in_unit == Int32(0):
                            sf_offset = self._compute_sf_offset(
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
            sf_col_idx = local_tidx // _threads_per_sf
            thread_in_unit = local_tidx % _threads_per_sf

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
                        # Fast path: padding row - zero out ALL scale factors
                        # Thread-stride loop since padded_sf_cols may exceed
                        # num_sf_blocks_per_row (when K/32 is not a multiple of 4)
                        if thread_in_unit == Int32(0):
                            pad_col = sf_col_idx
                            while pad_col < padded_sf_cols:
                                sf_offset = self._compute_sf_offset(
                                    row_idx, pad_col, padded_sf_cols
                                )
                                mScales[sf_offset] = Uint8(0)
                                pad_col = pad_col + num_sf_blocks_per_row
                    else:
                        # Normal path: process actual data
                        if sf_col_idx < num_sf_blocks_per_row:
                            elem_idx = (
                                sf_col_idx * SF_VEC_SIZE
                                + thread_in_unit * _elts_per_thread
                            )

                            row_input = mInput[row_idx, None]

                            if cutlass.const_expr(self.use_2t_per_sf):
                                if cutlass.const_expr(self.is_float32):
                                    # fp32: 16 elements = 2x256-bit loads
                                    (
                                        v0,
                                        v1,
                                        v2,
                                        v3,
                                        v4,
                                        v5,
                                        v6,
                                        v7,
                                    ) = ld_global_v8_u32(
                                        get_ptr_as_int64(row_input, elem_idx)
                                    )
                                    (
                                        v8,
                                        v9,
                                        v10,
                                        v11,
                                        v12,
                                        v13,
                                        v14,
                                        v15,
                                    ) = ld_global_v8_u32(
                                        get_ptr_as_int64(row_input, elem_idx + Int32(8))
                                    )
                                    local_max = fmax_f32(
                                        float_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7),
                                        float_max_abs_8(
                                            v8, v9, v10, v11, v12, v13, v14, v15
                                        ),
                                    )
                                else:
                                    input_ptr_lo = get_ptr_as_int64(row_input, elem_idx)
                                    input_ptr_hi = get_ptr_as_int64(
                                        row_input, elem_idx + Int32(8)
                                    )
                                    v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_lo)
                                    v4, v5, v6, v7 = ld_global_v4_u32(input_ptr_hi)
                                    if cutlass.const_expr(self.is_bfloat16):
                                        local_max = bfloat2_hmax_reduce_to_f32(
                                            bfloat2_max_abs_8(
                                                v0, v1, v2, v3, v4, v5, v6, v7
                                            )
                                        )
                                    else:
                                        local_max = hmax_reduce_to_f32(
                                            half2_max_abs_8(
                                                v0, v1, v2, v3, v4, v5, v6, v7
                                            )
                                        )
                                global_max = reduce_max_2threads(local_max)
                            else:
                                if cutlass.const_expr(self.is_float32):
                                    # fp32: 8 elements = 1x256-bit load
                                    (
                                        v0,
                                        v1,
                                        v2,
                                        v3,
                                        v4,
                                        v5,
                                        v6,
                                        v7,
                                    ) = ld_global_v8_u32(
                                        get_ptr_as_int64(row_input, elem_idx)
                                    )
                                    local_max = float_max_abs_8(
                                        v0, v1, v2, v3, v4, v5, v6, v7
                                    )
                                else:
                                    input_ptr_i64 = get_ptr_as_int64(
                                        row_input, elem_idx
                                    )
                                    v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_i64)
                                    if cutlass.const_expr(self.is_bfloat16):
                                        local_max = bfloat2_hmax_reduce_to_f32(
                                            bfloat2_max_abs_4(v0, v1, v2, v3)
                                        )
                                    else:
                                        local_max = hmax_reduce_to_f32(
                                            half2_max_abs_4(v0, v1, v2, v3)
                                        )
                                global_max = reduce_max_4threads(local_max)

                            inv_e4m3_max = Float32(INV_FLOAT8_E4M3_MAX)
                            scale_ue8m0_u32 = float_to_ue8m0_fast(
                                global_max * inv_e4m3_max
                            )
                            scale_ue8m0 = scale_ue8m0_u32.to(Uint8)
                            inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)

                            row_output = mOutput[row_idx, None]
                            if cutlass.const_expr(self.use_2t_per_sf):
                                if cutlass.const_expr(self.is_float32):
                                    fp8_lo = floatx8_to_fp8x8_packed(
                                        v0, v1, v2, v3, v4, v5, v6, v7, inv_scale
                                    )
                                    fp8_hi = floatx8_to_fp8x8_packed(
                                        v8, v9, v10, v11, v12, v13, v14, v15, inv_scale
                                    )
                                elif cutlass.const_expr(self.is_bfloat16):
                                    fp8_lo = bfloat2x4_to_fp8x8_packed(
                                        v0, v1, v2, v3, inv_scale
                                    )
                                    fp8_hi = bfloat2x4_to_fp8x8_packed(
                                        v4, v5, v6, v7, inv_scale
                                    )
                                else:
                                    fp8_lo = half2x4_to_fp8x8_packed(
                                        v0, v1, v2, v3, inv_scale
                                    )
                                    fp8_hi = half2x4_to_fp8x8_packed(
                                        v4, v5, v6, v7, inv_scale
                                    )
                                st_global_u64(
                                    get_ptr_as_int64(row_output, elem_idx), fp8_lo
                                )
                                st_global_u64(
                                    get_ptr_as_int64(row_output, elem_idx + Int32(8)),
                                    fp8_hi,
                                )
                            else:
                                if cutlass.const_expr(self.is_float32):
                                    fp8_packed = floatx8_to_fp8x8_packed(
                                        v0, v1, v2, v3, v4, v5, v6, v7, inv_scale
                                    )
                                elif cutlass.const_expr(self.is_bfloat16):
                                    fp8_packed = bfloat2x4_to_fp8x8_packed(
                                        v0, v1, v2, v3, inv_scale
                                    )
                                else:
                                    fp8_packed = half2x4_to_fp8x8_packed(
                                        v0, v1, v2, v3, inv_scale
                                    )
                                st_global_u64(
                                    get_ptr_as_int64(row_output, elem_idx), fp8_packed
                                )

                            if thread_in_unit == Int32(0):
                                sf_offset = self._compute_sf_offset(
                                    row_idx, sf_col_idx, padded_sf_cols
                                )
                                mScales[sf_offset] = scale_ue8m0

                        # Handle padding SF columns for this row
                        # Thread-stride loop starting from first padding column
                        if cutlass.const_expr(
                            self.num_sf_blocks_per_row != self.padded_sf_cols
                        ):
                            if thread_in_unit == Int32(0):
                                pad_col = num_sf_blocks_per_row + sf_col_idx
                                while pad_col < padded_sf_cols:
                                    sf_offset = self._compute_sf_offset(
                                        row_idx, pad_col, padded_sf_cols
                                    )
                                    mScales[sf_offset] = Uint8(0)
                                    pad_col = pad_col + num_sf_blocks_per_row

                row_batch_idx = row_batch_idx + grid_dim_x
                # Update row_idx for next iteration
                row_idx = row_batch_idx * rows_per_block + row_in_block

        # PDL: Signal that dependent kernels can start early
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# PyTorch Integration with TVM-FFI
# =============================================================================


# torch dtype -> (cache key, cutlass dtype) for the supported input dtypes.
_CUTLASS_DTYPE_MAP = {
    "fp16": cutlass.Float16,
    "bf16": cutlass.BFloat16,
    "fp32": cutlass.Float32,
}
_TORCH_DTYPE_KEY = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


@functools.cache
def _get_compiled_kernel_mxfp8_linear(
    dtype_key: str,
    K: int,
    enable_pdl: bool = False,
    use_2t_per_sf: bool = True,
) -> Tuple[Callable, int]:
    """
    Get or compile LINEAR layout kernel with TVM-FFI.

    Cached by (K, dtype, pdl, use_2t) - M-agnostic, device-independent
    compilation.

    Returns:
        Tuple of (compiled_kernel, sf_blocks_per_tb) where sf_blocks_per_tb
        is used by the caller to compute num_blocks at runtime.
    """
    cutlass_dtype = _CUTLASS_DTYPE_MAP[dtype_key]
    kernel_obj = MXFP8QuantizeLinearKernel(cutlass_dtype, K, enable_pdl, use_2t_per_sf)

    # Use symbolic M for dynamic batch sizes
    sym_m = cute.sym_int()

    # Create fake tensors for compilation
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K), stride_order=(1, 0), assumed_align=16
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
        Int32(1),  # Dummy total_sf_blocks
        Int32(1),  # Dummy num_blocks
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel, kernel_obj.SF_BLOCKS_PER_TB


# =============================================================================
# CuTe-DSL Kernel Class for the 128x4 Swizzled Layout — Tile Iteration (SM107)
# =============================================================================


class MXFP8QuantizeTile128x4Kernel:
    """
    MXFP8 quantization for the 128x4 scale layout, one 128-row x (4*G)-SF-column
    tile per CTA iteration.

    Why a tile kernel: in the row-based kernel a warp owns one row, so its
    1-byte scale stores land in 8 different 128 B lines per 32 SF blocks (4 B
    per line at a 512 B stride) -- as many L2 requests as the 1 KB of input it
    reads. A 128-row tile's scales are one contiguous 512*G B block in this
    layout, so they are staged in shared memory and written back with a few
    full-line stores. Loads stay coalesced: 2 lanes per SF block (16 elements
    each), 8*G lanes per row segment, 32/(8*G) rows per warp instruction.

    Rows are padded to 128 by the host; padding rows and padding scale columns
    get zero scales. Loads are clamped rather than predicated so the 2-lane
    max-reduce shuffle never diverges; only stores are predicated.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        K: int,
        enable_pdl: bool = False,
        col_groups: int = 2,
    ):
        self.is_bfloat16 = dtype == cutlass.BFloat16
        self.is_float32 = dtype == cutlass.Float32
        self.enable_pdl = enable_pdl
        self.K = K
        assert K % SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // SF_VEC_SIZE
        self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4
        self.col_groups = col_groups
        self.sf_cols_per_tile = 4 * col_groups
        self.lanes_per_row = THREADS_PER_SF * self.sf_cols_per_tile
        assert WARP_SIZE % self.lanes_per_row == 0, "tile must fit a warp"
        self.rows_per_warp = WARP_SIZE // self.lanes_per_row
        self.threads = _TILE128X4_THREADS
        self.warps = self.threads // WARP_SIZE
        self.rows_per_pass = self.warps * self.rows_per_warp
        assert ROW_TILE_SIZE % self.rows_per_pass == 0
        self.passes = ROW_TILE_SIZE // self.rows_per_pass
        self.scale_tile_bytes = ROW_TILE_SIZE * self.sf_cols_per_tile
        assert self.scale_tile_bytes % self.threads == 0
        self.col_tiles = (
            self.num_sf_blocks_per_row + self.sf_cols_per_tile - 1
        ) // self.sf_cols_per_tile

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
            block=[self.threads, 1, 1],
            max_number_threads=[self.threads, 1, 1],
            min_blocks_per_mp=_TILE128X4_BLOCKS_PER_SM,
            smem=self.scale_tile_bytes,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.jit
    def _quantize_16(self, row_input, elem_idx: Int32):
        """Load this lane's 16 elements, reduce the SF-block max with the
        partner lane, and return (scale_ue8m0, fp8_lo, fp8_hi)."""
        if cutlass.const_expr(self.is_float32):
            v0, v1, v2, v3, v4, v5, v6, v7 = ld_global_v8_u32(
                get_ptr_as_int64(row_input, elem_idx)
            )
            v8, v9, v10, v11, v12, v13, v14, v15 = ld_global_v8_u32(
                get_ptr_as_int64(row_input, elem_idx + Int32(8))
            )
            local_max = fmax_f32(
                float_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7),
                float_max_abs_8(v8, v9, v10, v11, v12, v13, v14, v15),
            )
        else:
            v0, v1, v2, v3 = ld_global_v4_u32(get_ptr_as_int64(row_input, elem_idx))
            v4, v5, v6, v7 = ld_global_v4_u32(
                get_ptr_as_int64(row_input, elem_idx + Int32(8))
            )
            if cutlass.const_expr(self.is_bfloat16):
                local_max = bfloat2_hmax_reduce_to_f32(
                    bfloat2_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7)
                )
            else:
                local_max = hmax_reduce_to_f32(
                    half2_max_abs_8(v0, v1, v2, v3, v4, v5, v6, v7)
                )
        global_max = reduce_max_2threads(local_max)
        normalized_max = global_max * Float32(INV_FLOAT8_E4M3_MAX)
        scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
        scale_ue8m0 = scale_ue8m0_u32.to(Uint8)
        inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)
        if cutlass.const_expr(self.is_float32):
            fp8_lo = floatx8_to_fp8x8_packed(v0, v1, v2, v3, v4, v5, v6, v7, inv_scale)
            fp8_hi = floatx8_to_fp8x8_packed(
                v8, v9, v10, v11, v12, v13, v14, v15, inv_scale
            )
        elif cutlass.const_expr(self.is_bfloat16):
            fp8_lo = bfloat2x4_to_fp8x8_packed(v0, v1, v2, v3, inv_scale)
            fp8_hi = bfloat2x4_to_fp8x8_packed(v4, v5, v6, v7, inv_scale)
        else:
            fp8_lo = half2x4_to_fp8x8_packed(v0, v1, v2, v3, inv_scale)
            fp8_hi = half2x4_to_fp8x8_packed(v4, v5, v6, v7, inv_scale)
        return scale_ue8m0, fp8_lo, fp8_hi

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        M: Int32,
        padded_M: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        smem = cutlass.utils.SmemAllocator()
        s_scales = smem.allocate_tensor(
            Uint8, cute.make_layout((self.scale_tile_bytes,)), byte_alignment=16
        )

        num_sf_blocks_per_row = Int32(self.num_sf_blocks_per_row)
        padded_sf_cols = Int32(self.padded_sf_cols)
        sf_cols_per_tile = self.sf_cols_per_tile
        lanes_per_row = self.lanes_per_row
        rows_per_warp = self.rows_per_warp
        rows_per_pass = self.rows_per_pass
        col_tiles = Int32(self.col_tiles)

        warp = tidx // WARP_SIZE
        lane = tidx % WARP_SIZE
        row_in_pass = warp * rows_per_warp + lane // lanes_per_row
        lane_in_row = lane % lanes_per_row
        sf_col_in_tile = lane_in_row // THREADS_PER_SF
        thread_in_sf = lane_in_row % THREADS_PER_SF
        # smem byte for this lane's SF block, without the row term:
        #   (r % 32) * 16 + (r // 32) * 4 + (c % 4) + (c // 4) * 512
        col_smem_off = (sf_col_in_tile % 4) + (sf_col_in_tile // 4) * 512

        num_tiles = (padded_M // ROW_TILE_SIZE) * col_tiles
        tile = bidx
        while tile < num_tiles:
            row_tile = tile // col_tiles
            col_tile = tile % col_tiles
            row0 = row_tile * ROW_TILE_SIZE
            sf_col0 = col_tile * sf_cols_per_tile
            sf_col = sf_col0 + sf_col_in_tile
            col_valid = sf_col < num_sf_blocks_per_row
            # Clamp for the unconditional load; stores are predicated.
            sf_col_load = cutlass.min(sf_col, num_sf_blocks_per_row - 1)
            elem_idx = sf_col_load * SF_VEC_SIZE + thread_in_sf * ELTS_PER_THREAD

            p = Int32(0)
            while p < self.passes:
                local_r = p * rows_per_pass + row_in_pass
                r = row0 + local_r
                row_valid = r < M
                r_load = cutlass.min(r, M - 1)
                row_input = mInput[r_load, None]
                scale_ue8m0, fp8_lo, fp8_hi = self._quantize_16(row_input, elem_idx)
                if row_valid and col_valid:
                    row_output = mOutput[r, None]
                    st_global_u64(get_ptr_as_int64(row_output, elem_idx), fp8_lo)
                    st_global_u64(
                        get_ptr_as_int64(row_output, elem_idx + Int32(8)), fp8_hi
                    )
                else:
                    scale_ue8m0 = Uint8(0)
                if thread_in_sf == Int32(0):
                    s_off = (local_r % 32) * 16 + (local_r // 32) * 4 + col_smem_off
                    s_scales[s_off] = scale_ue8m0
                p = p + 1

            cute.arch.barrier()
            # The tile's scales are contiguous from base; skip column groups past
            # padded_sf_cols (only a straddling last tile when col_groups > 1).
            base = compute_sf_index_swizzled_128x4_gpu(row0, sf_col0, padded_sf_cols)
            b = tidx
            while b < self.scale_tile_bytes:
                group_col0 = sf_col0 + (b // 512) * 4
                if group_col0 < padded_sf_cols:
                    mScales[base + b] = s_scales[b]
                b = b + self.threads
            cute.arch.barrier()
            tile = tile + grid_dim_x

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _get_compiled_kernel_mxfp8_tile128x4(
    dtype_key: str,
    K: int,
    enable_pdl: bool = False,
    col_groups: int = 2,
) -> Tuple[Callable, int]:
    """Get or compile the SM107 128x4 tile kernel. Returns (kernel, col_tiles)."""
    cutlass_dtype = _CUTLASS_DTYPE_MAP[dtype_key]
    kernel_obj = MXFP8QuantizeTile128x4Kernel(
        cutlass_dtype, K, enable_pdl, col_groups=col_groups
    )
    sym_m = cute.sym_int()
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K), stride_order=(1, 0), assumed_align=16
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
        Int32(1),  # Dummy num_blocks
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_kernel, kernel_obj.col_tiles


@functools.cache
def _get_compiled_kernel_mxfp8_swizzled(
    dtype_key: str,
    K: int,
    enable_pdl: bool = False,
    use_2t_per_sf: bool = True,
    sf_layout: int = SF_LAYOUT_128x4,
    small_blocks: bool = False,
) -> Tuple[Callable, int]:
    """
    Get or compile SWIZZLED layout kernel with TVM-FFI.

    Cached by (K, dtype, pdl, use_2t, sf_layout, small_blocks) - M-agnostic,
    device-independent compilation.

    Returns:
        Tuple of (compiled_kernel, rows_per_block) where rows_per_block
        is used by the caller to compute num_blocks at runtime.
    """
    cutlass_dtype = _CUTLASS_DTYPE_MAP[dtype_key]
    kernel_obj = MXFP8QuantizeSwizzledKernel(
        cutlass_dtype,
        K,
        enable_pdl,
        use_2t_per_sf,
        sf_layout=sf_layout,
        small_blocks=small_blocks,
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
        Int32(1),  # Dummy num_blocks
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel, kernel_obj.rows_per_block


@flashinfer_api
def mxfp8_quantize_cute_dsl(
    input: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    enable_pdl: bool | None = None,
    is_sf_8x4_layout: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Quantize input tensor to MXFP8 format using the CuTe-DSL kernel.

    GPU implementation with dual-path optimization:

    - **LINEAR layout**: SF-block based iteration (fast).
    - **SWIZZLED layout**: row-based iteration with a padding fast path.

    The kernel is compiled once per ``(K, dtype, pdl, sf_layout)`` tuple
    and handles varying ``M`` (batch size) at runtime without
    recompilation.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``[M, K]`` with dtype fp16/bf16/fp32.
    is_sf_swizzled_layout : bool
        Whether to use a swizzled layout (``True``) or linear (``False``).
        When ``True``, the layout is 128x4 by default; pass
        ``is_sf_8x4_layout=True`` for 8x4.
    alignment : int
        Alignment for the K dimension (default 32; must be a multiple of
        ``SF_VEC_SIZE``).
    enable_pdl : bool, optional
        Whether to enable Programmatic Dependent Launch.  Auto-detected
        from device capability (SM >= 9.0) when ``None``.
    is_sf_8x4_layout : bool
        When ``is_sf_swizzled_layout`` is ``True``, selects 8x4 swizzling
        instead of the default 128x4.  Ignored for the linear layout.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(fp8_tensor, scale_tensor)`` where ``fp8_tensor`` is the
        quantized tensor of shape ``[M, padded_K]`` with dtype
        ``float8_e4m3fn`` and ``scale_tensor`` is the UE8M0 scale-factor
        tensor (``uint8``).
    """
    from ...utils import device_support_pdl, get_compute_capability

    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"Input dtype must be float16, bfloat16, or float32, got {input.dtype}"
    )
    assert input.is_cuda, "Input must be on CUDA device"
    assert alignment % SF_VEC_SIZE == 0, (
        f"alignment must be divisible by SF_VEC_SIZE={SF_VEC_SIZE}"
    )

    # Auto-detect PDL support based on device capability.
    # If caller passes True explicitly, still check hardware support.
    enable_pdl = device_support_pdl(input.device) if enable_pdl is not False else False

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

    # The kernels use vectorized global loads: 128-bit (16B) for fp16/bf16 and
    # 256-bit (32B) for fp32. `.contiguous()` preserves contiguous *views* with
    # a storage offset (e.g. `pool[4:4 + m * k].view(m, k)`), whose data_ptr can
    # be under-aligned and would fault the vector load. Materialize an aligned
    # copy in that rare case (fresh torch allocations are 256B-aligned).
    required_align = 32 if input.dtype == torch.float32 else 16
    if input_padded.data_ptr() % required_align != 0:
        input_padded = input_padded.clone(memory_format=torch.contiguous_format)

    dtype_key = _TORCH_DTYPE_KEY[input.dtype]

    # Cached device-specific target grid for grid size computation
    is_sm107 = get_compute_capability(input.device) == (10, 7)
    small_blocks = (
        is_sm107
        and is_sf_swizzled_layout
        and _use_sm107_small_blocks(m, padded_k, input.dtype == torch.float32)
    )
    blocks_per_sm = _BLOCKS_PER_SM
    if (
        m * padded_k >= 1 << 20
        and (
            not is_sf_swizzled_layout
            or is_sf_8x4_layout
            or m >= 1024
            or m * padded_k >= 1 << 23
        )
        and is_sm107
        and not small_blocks
        # FP32 linear is neutral-to-slightly-worse with a lower cap
        # (0.96-1.02x in cold-L2 sweeps); leave it at the default.
        and not (not is_sf_swizzled_layout and input.dtype == torch.float32)
    ):
        # The swizzled kernel uses up to 1024 threads per CTA. On Rubin,
        # one persistent CTA per SM avoids extra waves; linear uses two.
        # Keep the original cap for medium 128x4 inputs: cold-L2 sweeps
        # show a crossover regression before 1024 rows / 8M elements.
        blocks_per_sm = 1 if is_sf_swizzled_layout else 2
    target_grid = get_num_sm(input.device) * blocks_per_sm

    # Compute M-dependent values outside the cached kernel
    num_sf_blocks_per_row = padded_k // SF_VEC_SIZE

    # Choose 2T/SF (optimized) vs 4T/SF (legacy) based on problem size.
    # 2T/SF doubles memory-level parallelism per warp but halves the grid,
    # so it only helps when there are enough SF blocks to fill all SMs.
    total_sf_blocks_for_dispatch = m * num_sf_blocks_per_row
    use_2t = total_sf_blocks_for_dispatch >= MXFP8_2T_SF_THRESHOLD

    if is_sf_swizzled_layout:
        # Swizzled layout: compute padded_M and scale_output_size
        sf_layout = SF_LAYOUT_8x4 if is_sf_8x4_layout else SF_LAYOUT_128x4
        row_tile_size = 8 if is_sf_8x4_layout else ROW_TILE_SIZE
        padded_m = ((m + row_tile_size - 1) // row_tile_size) * row_tile_size
        padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4
        scale_output_size = padded_m * padded_sf_cols

        use_tile = (
            is_sm107
            and not is_sf_8x4_layout
            and _use_sm107_tile128x4(m, padded_k, input.element_size())
        )
        if use_tile:
            # SM107 only: 128-row tiles with shared-memory-staged scales.
            kernel_fn, col_tiles = _get_compiled_kernel_mxfp8_tile128x4(
                dtype_key, padded_k, enable_pdl, _TILE128X4_COL_GROUPS
            )
            num_blocks = min(
                (padded_m // ROW_TILE_SIZE) * col_tiles,
                get_num_sm(input.device) * _TILE128X4_BLOCKS_PER_SM,
            )
        else:
            kernel_fn, rows_per_block = _get_compiled_kernel_mxfp8_swizzled(
                dtype_key,
                padded_k,
                enable_pdl,
                use_2t,
                sf_layout,
                small_blocks,
            )

            num_blocks = min(
                (padded_m + rows_per_block - 1) // rows_per_block, target_grid
            )

        fp8_output = torch.empty(m, padded_k, dtype=torch.uint8, device=input.device)
        scale_output = torch.empty(
            scale_output_size, dtype=torch.uint8, device=input.device
        )

        kernel_fn(input_padded, fp8_output, scale_output, m, padded_m, num_blocks)
    else:
        # Linear layout: compute total_sf_blocks
        total_sf_blocks = m * num_sf_blocks_per_row
        scale_output_size = total_sf_blocks

        kernel_fn, sf_blocks_per_tb = _get_compiled_kernel_mxfp8_linear(
            dtype_key, padded_k, enable_pdl, use_2t
        )

        num_blocks = min(
            (total_sf_blocks + sf_blocks_per_tb - 1) // sf_blocks_per_tb, target_grid
        )

        fp8_output = torch.empty(m, padded_k, dtype=torch.uint8, device=input.device)
        scale_output = torch.empty(
            scale_output_size, dtype=torch.uint8, device=input.device
        )

        kernel_fn(input_padded, fp8_output, scale_output, total_sf_blocks, num_blocks)

    fp8_tensor = fp8_output.view(torch.float8_e4m3fn)

    return fp8_tensor, scale_output


__all__ = [
    "SF_LAYOUT_128x4",
    "SF_LAYOUT_8x4",
    "SF_LAYOUT_LINEAR",
    "MXFP8QuantizeLinearKernel",
    "MXFP8QuantizeSwizzledKernel",
    "mxfp8_quantize_cute_dsl",
    "_get_compiled_kernel_mxfp8_linear",
    "_get_compiled_kernel_mxfp8_swizzled",
]
