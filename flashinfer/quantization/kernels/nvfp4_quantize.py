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

NVFP4 Quantization using CuTe-DSL
=================================

NVFP4 quantization kernel using CuTe-DSL.
Supports multiple scale factor layouts: swizzled 128x4, swizzled 8x4, and linear.

Key differences from MXFP4:
- sf_vec_size=16 (vs 32 for MXFP4)
- E4M3 scale factors (vs UE8M0 for MXFP4)
- User-provided global_scale (vs auto-computed for MXFP4)
"""

import functools
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import torch
from cutlass import Float32, Int32, Uint8

from ...api_logging import flashinfer_api
from ...cute_dsl.fp4_common import get_ptr_as_int64, st_global_u64
from ...cute_dsl.utils import get_num_sm
from ..quantization_cute_dsl_utils import (
    NVFP4_SF_VEC_SIZE,
    ROW_TILE_SIZE,
    compute_sf_index_swizzled_128x4_gpu,
    compute_sf_index_swizzled_8x4_gpu,
    compute_sf_index_linear_gpu,
    half2_max_abs_8 as half2_max_abs_8_fn,
    bfloat2_max_abs_8 as bfloat2_max_abs_8_fn,
    hmax_reduce_to_f32,
    bfloat2_hmax_reduce_to_f32,
    half2x8_to_e2m1x16_packed,
    bfloat2x8_to_e2m1x16_packed,
    process_nvfp4_block_half,
    process_nvfp4_block_bfloat,
    process_nvfp4_block_fp8,
)

SF_LAYOUT_128x4 = 0
SF_LAYOUT_8x4 = 1
SF_LAYOUT_LINEAR = 2

_BLOCKS_PER_SM = 4
_MAX_THREADS_PER_BLOCK = 1024
_MIN_THREADS = 128
_MAX_THREADS = 512
_DEFAULT_THREADS = 256


def _compute_optimal_threads_for_k(K: int) -> int:
    """
    Compute optimal thread count for 100% thread utilization.

    For NVFP4, each thread processes one SF block (16 elements).
    threads_per_row = K / 16 = num_sf_blocks_per_row

    We prefer LARGER thread counts (up to _MAX_THREADS) for better occupancy,
    while maintaining 100% thread utilization.
    """
    threads_per_row = K // NVFP4_SF_VEC_SIZE

    if threads_per_row >= _MAX_THREADS:
        return _MAX_THREADS

    if threads_per_row <= _MAX_THREADS:
        threads = (_MAX_THREADS // threads_per_row) * threads_per_row
        if threads >= _MIN_THREADS:
            return threads
        threads = threads_per_row
        while threads < _MIN_THREADS:
            threads += threads_per_row
        if threads <= _MAX_THREADS:
            return threads

    return _DEFAULT_THREADS


def _compute_swizzled_layout_sf_size(
    total_row: int, total_column: int, row_size: int = 128
) -> int:
    """Compute size of swizzled scale factor buffer."""
    padded_row = (total_row + row_size - 1) // row_size * row_size
    padded_column = (total_column + 3) // 4 * 4
    return padded_row * padded_column


# =============================================================================
# CuTe-DSL Kernel Class for NVFP4 Swizzled Layout
# =============================================================================


class NVFP4QuantizeSwizzledKernel:
    """
    NVFP4 quantization kernel supporting multiple scale factor layouts.

    Supported layouts:
    - 128x4 (swizzled): Optimized for GEMM with large tileN
    - 8x4 (swizzled): Optimized for GEMM with small tileN
    - linear: Simple row-major layout, no swizzling

    Key features:
    - E4M3 scale factors (FP8 format) with user-provided global_scale
    - sf_vec_size=16 (each thread processes 16 elements)
    - Multi-row processing when K is small, column loop when K is large
    - Row-based iteration with grid-stride loop
    - Padding row fast path for zeroing scale factors

    This kernel is M-agnostic: compiled once per (K, dtype, sf_layout, pdl)
    combination. M-dependent values (M, padded_M) and global_scale are passed
    at runtime.
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
        self.is_fp8 = dtype == cutlass.Float8E4M3FN
        self.enable_pdl = enable_pdl
        self.sf_layout = sf_layout
        self.sf_is_128x4 = sf_layout == SF_LAYOUT_128x4
        self.sf_is_8x4 = sf_layout == SF_LAYOUT_8x4

        assert K % NVFP4_SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // NVFP4_SF_VEC_SIZE

        if sf_layout == SF_LAYOUT_LINEAR:
            self.padded_sf_cols = self.num_sf_blocks_per_row
            self.row_tile_size = 1
        elif sf_layout == SF_LAYOUT_8x4:
            self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4
            self.row_tile_size = 8
        else:
            self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4
            self.row_tile_size = ROW_TILE_SIZE  # 128

        self.num_threads = _compute_optimal_threads_for_k(K)

        self.threads_per_row = self.num_sf_blocks_per_row

        if self.threads_per_row <= self.num_threads:
            self.rows_per_block = self.num_threads // self.threads_per_row
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
            if cutlass.const_expr(self.sf_is_8x4):
                return compute_sf_index_swizzled_8x4_gpu(row_idx, col_idx, padded_cols)
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
        num_blocks: Int32,
        mGlobalScale: cute.Tensor,
        stream,
    ):
        threads_per_block = self.num_threads

        self.kernel(mInput, mOutput, mScales, M, padded_M, mGlobalScale).launch(
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
        mGlobalScale: cute.Tensor,
    ):
        """
        NVFP4 quantization kernel with swizzled scale factor layout.

        Dual-path kernel with compile-time selection:
        - Small K path: Multi-row processing for improved thread utilization
        - Large K path: Single row with column loop

        Each thread processes one SF block (16 elements):
        1. Load 16 elements (2 x 128-bit for fp16/bf16, 1 x 128-bit for fp8)
        2. Compute max absolute value using SIMD reduction
        3. Compute E4M3 scale: cvt_f32_to_e4m3(global_scale * max / 6.0)
        4. Store scale factor using layout-specific indexing
        5. Back-convert E4M3, compute output_scale = global_scale / scale_back
        6. Scale elements and convert to E2M1
        7. Store 8 bytes (16 FP4 values)
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        # Read global_scale from device memory (avoids CPU-GPU sync at launch)
        global_scale = Float32(mGlobalScale[Int32(0)])

        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        padded_sf_cols = self.padded_sf_cols
        num_threads = self.num_threads
        rows_per_block = self.rows_per_block
        threads_per_row = self.threads_per_row

        if cutlass.const_expr(not self.needs_col_loop):
            # ===== SMALL K PATH: Multi-row processing =====
            row_in_block = tidx // threads_per_row
            sf_idx_in_row = tidx % threads_per_row

            row_batch_idx = bidx
            total_row_batches = cute.ceil_div(padded_M, rows_per_block)

            while row_batch_idx < total_row_batches:
                base_row = row_batch_idx * rows_per_block
                row_idx = base_row + row_in_block

                if row_idx < padded_M:
                    is_padding_row = row_idx >= M

                    if is_padding_row:
                        local_sf_idx = sf_idx_in_row
                        while local_sf_idx < padded_sf_cols:
                            sf_offset = self._compute_sf_offset(
                                row_idx, local_sf_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)
                            local_sf_idx = local_sf_idx + threads_per_row
                    else:
                        if sf_idx_in_row < num_sf_blocks_per_row:
                            elem_base = sf_idx_in_row * NVFP4_SF_VEC_SIZE
                            row_input = mInput[row_idx, None]

                            if cutlass.const_expr(self.is_fp8):
                                scale_fp8, packed64 = process_nvfp4_block_fp8(
                                    row_input, elem_base, global_scale
                                )
                            elif cutlass.const_expr(self.is_bfloat16):
                                scale_fp8, packed64 = process_nvfp4_block_bfloat(
                                    row_input, elem_base, global_scale
                                )
                            else:
                                scale_fp8, packed64 = process_nvfp4_block_half(
                                    row_input, elem_base, global_scale
                                )

                            sf_offset = self._compute_sf_offset(
                                row_idx, sf_idx_in_row, padded_sf_cols
                            )
                            mScales[sf_offset] = scale_fp8

                            row_output = mOutput[row_idx, None]
                            out_base = sf_idx_in_row * (NVFP4_SF_VEC_SIZE // 2)
                            out_ptr = get_ptr_as_int64(row_output, out_base)
                            st_global_u64(out_ptr, packed64)

                        padding_sf_start = num_sf_blocks_per_row + sf_idx_in_row
                        while padding_sf_start < padded_sf_cols:
                            sf_offset = self._compute_sf_offset(
                                row_idx, padding_sf_start, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)
                            padding_sf_start = padding_sf_start + threads_per_row

                row_batch_idx = row_batch_idx + grid_dim_x

        else:
            # ===== LARGE K PATH: Single row with column loop =====
            row_idx = bidx
            while row_idx < padded_M:
                is_padding_row = row_idx >= M

                sf_idx = Int32(tidx)

                if is_padding_row:
                    while sf_idx < padded_sf_cols:
                        sf_offset = self._compute_sf_offset(
                            row_idx, sf_idx, padded_sf_cols
                        )
                        mScales[sf_offset] = Uint8(0)
                        sf_idx = sf_idx + num_threads
                else:
                    num_sf_iters = (
                        num_sf_blocks_per_row + num_threads - 1
                    ) // num_threads

                    for sf_iter in range(num_sf_iters):
                        local_sf_idx = sf_iter * num_threads + tidx

                        if local_sf_idx < num_sf_blocks_per_row:
                            elem_base = local_sf_idx * NVFP4_SF_VEC_SIZE
                            row_input = mInput[row_idx, None]

                            if cutlass.const_expr(self.is_fp8):
                                scale_fp8, packed64 = process_nvfp4_block_fp8(
                                    row_input, elem_base, global_scale
                                )
                            elif cutlass.const_expr(self.is_bfloat16):
                                scale_fp8, packed64 = process_nvfp4_block_bfloat(
                                    row_input, elem_base, global_scale
                                )
                            else:
                                scale_fp8, packed64 = process_nvfp4_block_half(
                                    row_input, elem_base, global_scale
                                )

                            sf_offset = self._compute_sf_offset(
                                row_idx, local_sf_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = scale_fp8

                            row_output = mOutput[row_idx, None]
                            out_base = local_sf_idx * (NVFP4_SF_VEC_SIZE // 2)
                            out_ptr = get_ptr_as_int64(row_output, out_base)
                            st_global_u64(out_ptr, packed64)

                    padding_sf_start = num_sf_blocks_per_row + tidx
                    while padding_sf_start < padded_sf_cols:
                        sf_offset = self._compute_sf_offset(
                            row_idx, padding_sf_start, padded_sf_cols
                        )
                        mScales[sf_offset] = Uint8(0)
                        padding_sf_start = padding_sf_start + num_threads

                row_idx = row_idx + grid_dim_x

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# CuTe-DSL TMA Kernel Class for NVFP4
# =============================================================================

_TMA_ROW_TILE = 16
_TMA_COL_TILE = 64  # Per-warp column tile
_TMA_NUM_CONSUMER_WARPS = 8
_TMA_NUM_STAGES = 4
_TMA_COLS_PER_STAGE = _TMA_NUM_CONSUMER_WARPS * _TMA_COL_TILE  # 512


def _round_up(x: int, d: int) -> int:
    return ((x + d - 1) // d) * d


class NVFP4QuantizeTMAKernel:
    """
    TMA-based NVFP4 quantization kernel with pipelined producer-consumer
    warp specialization, matching the CUDA TMA kernel architecture.

    Architecture (matches csrc/nv_internal/.../quantization.cuh):
    - 1 producer warp (warp 0) issues TMA G2S loads into staged SMEM buffers
    - 8 consumer warps (warps 1-8) read from SMEM, quantize, write to GMEM
    - PipelineTmaAsync manages multi-stage buffering (4 stages)
    - Each TMA tile: [16, 512] = 16 rows x 8 warps x 64 cols per warp
    - Each consumer warp: 4 threads/row x 8 rows/warp, 2 row iterations
    - Each thread: 16 elements (1 SF block) via 2 x ld.shared.v4.u32
    - Grid-stride loop over row tiles, inner loop over K/512 col chunks

    Effective when M >= 1024 and K is a multiple of 512.
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
        self.is_fp8 = dtype == cutlass.Float8E4M3FN
        self.enable_pdl = enable_pdl
        self.sf_layout = sf_layout
        self.sf_is_128x4 = sf_layout == SF_LAYOUT_128x4
        self.sf_is_8x4 = sf_layout == SF_LAYOUT_8x4

        assert not self.is_fp8, "FP8 input not yet supported for TMA kernel"
        assert K % _TMA_COLS_PER_STAGE == 0, (
            f"K ({K}) must be a multiple of {_TMA_COLS_PER_STAGE} for TMA kernel"
        )

        self.num_sf_blocks_per_row = K // NVFP4_SF_VEC_SIZE
        self.num_col_chunks = K // _TMA_COLS_PER_STAGE

        if sf_layout == SF_LAYOUT_LINEAR:
            self.padded_sf_cols = self.num_sf_blocks_per_row
            self.row_tile_size = 1
        elif sf_layout == SF_LAYOUT_8x4:
            self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4
            self.row_tile_size = 8
        else:
            self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4
            self.row_tile_size = ROW_TILE_SIZE

        self.num_consumer_warps = _TMA_NUM_CONSUMER_WARPS  # 8
        self.num_stages = _TMA_NUM_STAGES
        self.producer_warp_id = 0  # Warp 0 is producer (matches CUDA kernel)
        self.threads_per_cta = 32 * (self.num_consumer_warps + 1)  # 288
        self.rows_per_block = _TMA_ROW_TILE
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)
        self.elems_per_stage = _TMA_ROW_TILE * _TMA_COLS_PER_STAGE  # 8192

        # Thread indexing constants (matches CUDA TmaKernelTraitsTwoBytes)
        self.THREADS_PER_ROW = 4  # laneIdx % 4
        self.ROWS_PER_WARP = 8  # 32 / 4
        self.ROW_ITERATIONS = _TMA_ROW_TILE // self.ROWS_PER_WARP  # 2
        self.ELTS_PER_THREAD = NVFP4_SF_VEC_SIZE  # 16

    @cute.jit
    def _compute_sf_offset(
        self, row_idx: Int32, col_idx: Int32, padded_cols: Int32
    ) -> Int32:
        if cutlass.const_expr(self.sf_is_128x4):
            return compute_sf_index_swizzled_128x4_gpu(row_idx, col_idx, padded_cols)
        else:
            if cutlass.const_expr(self.sf_is_8x4):
                return compute_sf_index_swizzled_8x4_gpu(row_idx, col_idx, padded_cols)
            else:
                return compute_sf_index_linear_gpu(row_idx, col_idx, padded_cols)

    @cute.jit
    def _quantize_sf_block(
        self,
        h0: cutlass.Uint32,
        h1: cutlass.Uint32,
        h2: cutlass.Uint32,
        h3: cutlass.Uint32,
        h4: cutlass.Uint32,
        h5: cutlass.Uint32,
        h6: cutlass.Uint32,
        h7: cutlass.Uint32,
        global_row: Int32,
        sf_col: Int32,
        global_scale: Float32,
        M: Int32,
        padded_M: Int32,
        padded_sf_cols: Int32,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
    ):
        """Quantize one 16-element SF block and write results to GMEM."""
        from ...cute_dsl.fp4_common import (
            cvt_f32_to_e4m3,
            nvfp4_compute_output_scale,
            rcp_approx_ftz,
        )

        if global_row < padded_M:
            is_padding_row = global_row >= M

            if is_padding_row:
                sf_offset = self._compute_sf_offset(global_row, sf_col, padded_sf_cols)
                mScales[sf_offset] = Uint8(0)
            else:
                if cutlass.const_expr(self.is_bfloat16):
                    block_max_h2 = bfloat2_max_abs_8_fn(h0, h1, h2, h3, h4, h5, h6, h7)
                    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
                else:
                    block_max_h2 = half2_max_abs_8_fn(h0, h1, h2, h3, h4, h5, h6, h7)
                    block_max = hmax_reduce_to_f32(block_max_h2)

                fp4_max_rcp = rcp_approx_ftz(Float32(6.0))
                scale_float = global_scale * (block_max * fp4_max_rcp)
                scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))

                output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

                if cutlass.const_expr(self.is_bfloat16):
                    packed64 = bfloat2x8_to_e2m1x16_packed(
                        h0, h1, h2, h3, h4, h5, h6, h7, output_scale
                    )
                else:
                    packed64 = half2x8_to_e2m1x16_packed(
                        h0, h1, h2, h3, h4, h5, h6, h7, output_scale
                    )

                sf_offset = self._compute_sf_offset(global_row, sf_col, padded_sf_cols)
                mScales[sf_offset] = scale_fp8

                row_output = mOutput[global_row, None]
                out_base = sf_col * Int32(NVFP4_SF_VEC_SIZE // 2)
                out_ptr = get_ptr_as_int64(row_output, out_base)
                st_global_u64(out_ptr, packed64)

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        M: Int32,
        padded_M: Int32,
        num_blocks: Int32,
        mGlobalScale: cute.Tensor,
        stream,
    ):
        # 3D global tensor: [padded_M, K/64, 64] so each warp's 64-col
        # stripe is the contiguous innermost dimension, matching the CUDA
        # TMA kernel's 3D tensor map.
        gInput = cute.make_tensor(
            mInput.iterator,
            cute.make_layout(
                (padded_M, self.K // _TMA_COL_TILE, _TMA_COL_TILE),
                stride=(self.K, _TMA_COL_TILE, 1),
            ),
        )

        # SMEM layout per stage: [rows=16, warps=8, cols_per_warp=64]
        # with SWIZZLE_128B applied.  Within each warp's [16, 64] tile the
        # row stride is 64 elems = 128 bytes, putting row bits in the S=3
        # range of the swizzle so different rows map to different banks.
        smem_swizzle = cute.make_swizzle(3, 4, 3)  # SWIZZLE_128B for 2B types
        smem_outer_single = cute.make_layout(
            (_TMA_ROW_TILE, _TMA_NUM_CONSUMER_WARPS, _TMA_COL_TILE),
            stride=(_TMA_COL_TILE, _TMA_ROW_TILE * _TMA_COL_TILE, 1),
        )
        smem_single_composed = cute.make_composed_layout(
            smem_swizzle, 0, smem_outer_single
        )

        cta_tiler = (_TMA_ROW_TILE, _TMA_NUM_CONSUMER_WARPS, _TMA_COL_TILE)
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gInput,
            smem_single_composed,
            cta_tiler,
        )

        total_smem_elems = self.elems_per_stage * self.num_stages
        # Staged outer layout (no swizzle — swizzle passed separately)
        smem_outer_staged = cute.make_layout(
            (_TMA_ROW_TILE, _TMA_NUM_CONSUMER_WARPS, _TMA_COL_TILE, self.num_stages),
            stride=(
                _TMA_COL_TILE,
                _TMA_ROW_TILE * _TMA_COL_TILE,
                1,
                self.elems_per_stage,
            ),
        )
        # Flat layout for manual-swizzle consumer reads
        smem_layout_flat = cute.make_layout((total_smem_elems,))

        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_outer_single)

        @cute.struct
        class SharedStorage:
            load_full_mbar: cute.struct.MemRange[cutlass.Int64, self.num_stages]
            load_empty_mbar: cute.struct.MemRange[cutlass.Int64, self.num_stages]
            smem_data: cute.struct.Align[
                cute.struct.MemRange[self.dtype, total_smem_elems],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (1,)
        )

        self.kernel(
            tma_atom,
            tma_tensor,
            mOutput,
            mScales,
            M,
            padded_M,
            mGlobalScale,
            smem_outer_staged,
            smem_swizzle,
            smem_layout_flat,
            cluster_layout_vmnk,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            max_number_threads=[
                self.threads_per_cta,
                1,
                1,
            ],  # __launch_bounds__(288, 2)
            min_blocks_per_mp=2,
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom: cute.CopyAtom,
        gInput_tma: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        M: Int32,
        padded_M: Int32,
        mGlobalScale: cute.Tensor,
        smem_outer_staged: cute.Layout,
        smem_swizzle: cute.Swizzle,
        smem_layout_flat: cute.Layout,
        cluster_layout_vmnk: cute.Layout,
    ):
        from ...cute_dsl.fp4_common import (
            get_smem_ptr_as_int32,
            ld_shared_v4_u32,
        )

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = tidx % 32

        global_scale = Float32(mGlobalScale[Int32(0)])
        padded_sf_cols = self.padded_sf_cols
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        num_col_chunks = self.num_col_chunks
        elems_per_stage = self.elems_per_stage

        # ---- SMEM allocation ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mbar_ptr = storage.load_full_mbar.data_ptr()
        # Swizzled tensor for TMA partition (address-space-correct writes)
        sData_staged = storage.smem_data.get_tensor(
            smem_outer_staged, swizzle=smem_swizzle
        )
        # Flat tensor for manual-swizzle consumer reads
        sData_flat = storage.smem_data.get_tensor(smem_layout_flat)

        # ---- Pipeline setup ----
        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        load_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )

        load_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar_ptr,
            num_stages=self.num_stages,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ---- TMA partition (3D: rows × warps × cols_per_warp) ----
        gSrc_tiled = cute.local_tile(
            gInput_tma,
            (_TMA_ROW_TILE, _TMA_NUM_CONSUMER_WARPS, _TMA_COL_TILE),
            (None, None, None),
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sData_staged, 0, 3),  # Group 3 tile modes
            cute.group_modes(gSrc_tiled, 0, 3),
        )

        num_row_tiles = cute.ceil_div(padded_M, _TMA_ROW_TILE)

        # ---- Consumer thread indexing (matches CUDA TmaKernelTraitsTwoBytes) ----
        # 4 threads per row, 8 rows per warp, 2 row iterations per stage
        col_idx_local = lane_idx % Int32(self.THREADS_PER_ROW)
        row_idx_local = lane_idx // Int32(self.THREADS_PER_ROW)

        # ======== Producer warp (warp 0) ========
        if warp_idx == self.producer_warp_id:
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )

            row_tile_idx = bidx
            while row_tile_idx < num_row_tiles:
                col_chunk = Int32(0)
                while col_chunk < num_col_chunks:
                    load_pipeline.producer_acquire(producer_state)

                    cute.copy(
                        tma_atom,
                        tAgA[(None, row_tile_idx, col_chunk, 0)],
                        tAsA[(None, producer_state.index)],
                        tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                    )

                    producer_state.advance()
                    col_chunk = col_chunk + Int32(1)

                row_tile_idx = row_tile_idx + grid_dim_x

            load_pipeline.producer_tail(producer_state)

        # ======== Consumer warps (warps 1-8) ========
        if warp_idx > 0:
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )

            # 0-indexed consumer warp id
            consumer_warp_idx = warp_idx - Int32(1)

            # Pre-compute warp tile base offset (constant per warp)
            # SMEM 3D layout: [rows=16, warps=8, cols=64] per stage
            # stride: [64, 1024, 1] → warp tile base = warp * 1024
            warp_tile_elems = _TMA_ROW_TILE * _TMA_COL_TILE  # 1024
            warp_tile_base = consumer_warp_idx * Int32(warp_tile_elems)

            # Float4 base position for this thread (0, 2, 4, 6)
            f4_base = col_idx_local * Int32(2)

            # Global column offset for SF index: warp's column within K
            base_col_in_stage = consumer_warp_idx * Int32(
                _TMA_COL_TILE
            ) + col_idx_local * Int32(self.ELTS_PER_THREAD)

            row_tile_idx = bidx
            while row_tile_idx < num_row_tiles:
                base_row = row_tile_idx * Int32(_TMA_ROW_TILE)

                col_chunk = Int32(0)
                while col_chunk < num_col_chunks:
                    load_pipeline.consumer_wait(consumer_state)
                    stage = consumer_state.index

                    # ---- Read ALL SMEM data with SWIZZLE_128B addressing ----
                    # Within each warp's [16,64] tile, the XOR pattern matches
                    # CUDA's load_input_vec: float4_idx ^= row & 7
                    # Physical elem offset in warp tile for (row, float4 f):
                    #   row * 64 + (f ^ (row & 7)) * 8
                    stage_base = stage * Int32(elems_per_stage)

                    # Row iteration 0 (row_idx_local = 0..7)
                    r0_xor = row_idx_local & Int32(7)
                    r0_f4_0 = f4_base ^ r0_xor
                    r0_f4_1 = (f4_base + Int32(1)) ^ r0_xor
                    r0_row_base = (
                        stage_base
                        + warp_tile_base
                        + row_idx_local * Int32(_TMA_COL_TILE)
                    )
                    r0_addr_0 = get_smem_ptr_as_int32(
                        sData_flat, r0_row_base + r0_f4_0 * Int32(8)
                    )
                    r0_addr_1 = get_smem_ptr_as_int32(
                        sData_flat, r0_row_base + r0_f4_1 * Int32(8)
                    )
                    r0_h0, r0_h1, r0_h2, r0_h3 = ld_shared_v4_u32(r0_addr_0)
                    r0_h4, r0_h5, r0_h6, r0_h7 = ld_shared_v4_u32(r0_addr_1)

                    # Row iteration 1 (row = row_idx_local + 8)
                    r1_row = row_idx_local + Int32(self.ROWS_PER_WARP)
                    r1_xor = r1_row & Int32(7)
                    r1_f4_0 = f4_base ^ r1_xor
                    r1_f4_1 = (f4_base + Int32(1)) ^ r1_xor
                    r1_row_base = (
                        stage_base + warp_tile_base + r1_row * Int32(_TMA_COL_TILE)
                    )
                    r1_addr_0 = get_smem_ptr_as_int32(
                        sData_flat, r1_row_base + r1_f4_0 * Int32(8)
                    )
                    r1_addr_1 = get_smem_ptr_as_int32(
                        sData_flat, r1_row_base + r1_f4_1 * Int32(8)
                    )
                    r1_h0, r1_h1, r1_h2, r1_h3 = ld_shared_v4_u32(r1_addr_0)
                    r1_h4, r1_h5, r1_h6, r1_h7 = ld_shared_v4_u32(r1_addr_1)

                    # ---- Release pipeline early (all data in registers) ----
                    load_pipeline.consumer_release(consumer_state)
                    consumer_state.advance()

                    # ---- Quantize and write: both row iterations ----
                    # Global column base for SF index computation
                    global_col_base = col_chunk * Int32(_TMA_COLS_PER_STAGE)
                    sf_col = (global_col_base + base_col_in_stage) // Int32(
                        NVFP4_SF_VEC_SIZE
                    )

                    # Row iteration 0
                    global_row_0 = base_row + row_idx_local
                    self._quantize_sf_block(
                        r0_h0,
                        r0_h1,
                        r0_h2,
                        r0_h3,
                        r0_h4,
                        r0_h5,
                        r0_h6,
                        r0_h7,
                        global_row_0,
                        sf_col,
                        global_scale,
                        M,
                        padded_M,
                        padded_sf_cols,
                        mOutput,
                        mScales,
                    )

                    # Row iteration 1
                    global_row_1 = base_row + row_idx_local + Int32(self.ROWS_PER_WARP)
                    self._quantize_sf_block(
                        r1_h0,
                        r1_h1,
                        r1_h2,
                        r1_h3,
                        r1_h4,
                        r1_h5,
                        r1_h6,
                        r1_h7,
                        global_row_1,
                        sf_col,
                        global_scale,
                        M,
                        padded_M,
                        padded_sf_cols,
                        mOutput,
                        mScales,
                    )

                    col_chunk = col_chunk + Int32(1)

                # Zero padding SF columns for swizzled layouts
                if cutlass.const_expr(self.sf_layout != SF_LAYOUT_LINEAR):
                    consumer_tid = (warp_idx - Int32(1)) * Int32(32) + lane_idx
                    if consumer_tid < _TMA_ROW_TILE:
                        pad_row_idx = base_row + consumer_tid
                        if pad_row_idx < padded_M:
                            padding_sf = Int32(num_sf_blocks_per_row)
                            while padding_sf < padded_sf_cols:
                                sf_offset = self._compute_sf_offset(
                                    pad_row_idx, padding_sf, padded_sf_cols
                                )
                                mScales[sf_offset] = Uint8(0)
                                padding_sf = padding_sf + Int32(1)

                row_tile_idx = row_tile_idx + grid_dim_x

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# PyTorch Integration with TVM-FFI
# =============================================================================


@functools.cache
def _get_compiled_kernel_nvfp4(
    dtype_key: str,
    K: int,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool = False,
) -> Tuple[Callable, int]:
    """
    Get or compile NVFP4 kernel with TVM-FFI.

    Cached by (K, dtype_key, sf_layout, pdl) - M-agnostic, device-independent
    compilation.

    Args:
        dtype_key: One of "float16", "bfloat16", "float8_e4m3fn".

    Returns:
        Tuple of (compiled_kernel, rows_per_block) where rows_per_block
        is used by the caller to compute num_blocks at runtime.
    """
    _dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    cutlass_dtype = _dtype_map[dtype_key]
    kernel_obj = NVFP4QuantizeSwizzledKernel(
        cutlass_dtype, K, sf_layout=sf_layout, enable_pdl=enable_pdl
    )

    sym_m = cute.sym_int()

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
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
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
        global_scale_fake,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel, kernel_obj.rows_per_block


_TMA_MIN_M = 1024


def _should_use_tma(m: int, k: int, dtype: torch.dtype) -> bool:
    """Determine if TMA kernel should be used based on problem dimensions."""
    if dtype == torch.float8_e4m3fn:
        return False
    return m >= _TMA_MIN_M and k % _TMA_COLS_PER_STAGE == 0


@functools.cache
def _get_compiled_kernel_nvfp4_tma(
    dtype_key: str,
    K: int,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool = False,
) -> Tuple[Callable, int]:
    """
    Get or compile TMA-based NVFP4 kernel with TVM-FFI.

    Cached by (K, dtype_key, sf_layout, pdl).
    """
    _dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
    }
    cutlass_dtype = _dtype_map[dtype_key]
    kernel_obj = NVFP4QuantizeTMAKernel(
        cutlass_dtype, K, sf_layout=sf_layout, enable_pdl=enable_pdl
    )

    sym_m = cute.sym_int()
    sym_padded_m = cute.sym_int()

    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_padded_m, K), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K // 2), stride_order=(1, 0), assumed_align=16
    )
    sym_scale_size = cute.sym_int()
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_scale_size,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        input_fake,
        output_fake,
        scales_fake,
        Int32(1),  # Dummy M
        Int32(1024),  # Dummy padded_M
        Int32(1),  # Dummy num_blocks
        global_scale_fake,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel, kernel_obj.rows_per_block


@flashinfer_api
def nvfp4_quantize_cute_dsl(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to NVFP4 format using CuTe-DSL kernel.

    This is a GPU implementation matching FlashInfer's nvfp4_quantize() behavior:
    - E4M3 scale factors (FP8)
    - E2M1 output format (4-bit, 2 values per byte)
    - Supports 128x4, 8x4, and linear scale factor layouts
    - sf_vec_size=16

    The kernel is compiled once per (K, dtype, sf_layout, pdl) combination and
    handles varying M (batch size) at runtime without recompilation.

    Args:
        input: Input tensor of shape [M, K] with dtype fp16/bf16/float8_e4m3fn
        global_scale: Scalar tensor (float32) for NVFP4 global scale factor
        sf_layout: Scale factor layout (0=128x4, 1=8x4, 2=linear).
        enable_pdl: Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability (SM >= 9.0).

    Returns:
        Tuple of:
            - fp4_tensor: Quantized tensor of shape [M, K/2] with dtype uint8
            - scale_tensor: E4M3 scale factors as uint8 tensor
              reshaped to [padded_rows, K/16]
    """
    from ...utils import device_support_pdl

    _supported_dtypes = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
    assert input.dtype in _supported_dtypes, (
        f"Input dtype must be one of {_supported_dtypes}, got {input.dtype}"
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

    assert k % NVFP4_SF_VEC_SIZE == 0, (
        f"K ({k}) must be divisible by NVFP4_SF_VEC_SIZE={NVFP4_SF_VEC_SIZE}"
    )

    input = input.contiguous()

    _torch_to_dtype_key = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3fn",
    }
    dtype_key = _torch_to_dtype_key[input.dtype]

    if isinstance(global_scale, torch.Tensor):
        global_scale_tensor = global_scale.float().reshape(1).contiguous()
        if not global_scale_tensor.is_cuda:
            global_scale_tensor = global_scale_tensor.to(input.device)
    else:
        global_scale_tensor = torch.tensor(
            [float(global_scale)], dtype=torch.float32, device=input.device
        )

    num_sm = get_num_sm(input.device)

    num_sf_blocks_per_row = k // NVFP4_SF_VEC_SIZE

    use_tma = _should_use_tma(m, k, input.dtype)

    if use_tma:
        tma_row_tile = _TMA_ROW_TILE
        if sf_layout == SF_LAYOUT_LINEAR:
            padded_m = _round_up(m, tma_row_tile)
            padded_sf_cols = num_sf_blocks_per_row
        elif sf_layout == SF_LAYOUT_8x4:
            padded_m = _round_up(m, max(tma_row_tile, 8))
            padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4
        else:
            padded_m = _round_up(m, max(tma_row_tile, ROW_TILE_SIZE))
            padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4

        scale_output_size = padded_m * padded_sf_cols

        kernel_fn, rows_per_block = _get_compiled_kernel_nvfp4_tma(
            dtype_key, k, sf_layout, enable_pdl
        )

        # Match CUDA TMA kernel: grid = min(row_tiles, SM_count * 2)
        tma_target_grid = num_sm * 2
        num_blocks = min(
            (padded_m + rows_per_block - 1) // rows_per_block, tma_target_grid
        )

        input_padded = input
        if padded_m > m:
            input_padded = torch.zeros(
                padded_m, k, dtype=input.dtype, device=input.device
            )
            input_padded[:m, :] = input

        fp4_output = torch.empty(m, k // 2, dtype=torch.uint8, device=input.device)
        scale_output = torch.empty(
            scale_output_size, dtype=torch.uint8, device=input.device
        )

        kernel_fn(
            input_padded,
            fp4_output,
            scale_output,
            m,
            padded_m,
            num_blocks,
            global_scale_tensor,
        )

        if sf_layout == SF_LAYOUT_LINEAR:
            scale_output = scale_output[: m * num_sf_blocks_per_row]

        scale_output = scale_output.reshape(-1, num_sf_blocks_per_row)

        return fp4_output, scale_output

    # Non-TMA path
    if sf_layout == SF_LAYOUT_LINEAR:
        row_tile_size = 1
        padded_m = m
        padded_sf_cols = num_sf_blocks_per_row
    elif sf_layout == SF_LAYOUT_8x4:
        row_tile_size = 8
        padded_m = ((m + row_tile_size - 1) // row_tile_size) * row_tile_size
        padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4
    else:
        row_tile_size = ROW_TILE_SIZE  # 128
        padded_m = ((m + row_tile_size - 1) // row_tile_size) * row_tile_size
        padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4

    scale_output_size = padded_m * padded_sf_cols

    kernel_fn, rows_per_block = _get_compiled_kernel_nvfp4(
        dtype_key, k, sf_layout, enable_pdl
    )

    default_target_grid = num_sm * _BLOCKS_PER_SM
    num_blocks = min(
        (padded_m + rows_per_block - 1) // rows_per_block, default_target_grid
    )

    fp4_output = torch.empty(m, k // 2, dtype=torch.uint8, device=input.device)
    scale_output = torch.empty(
        scale_output_size, dtype=torch.uint8, device=input.device
    )

    kernel_fn(
        input, fp4_output, scale_output, m, padded_m, num_blocks, global_scale_tensor
    )

    scale_output = scale_output.reshape(-1, num_sf_blocks_per_row)

    return fp4_output, scale_output


__all__ = [
    "SF_LAYOUT_128x4",
    "SF_LAYOUT_8x4",
    "SF_LAYOUT_LINEAR",
    "NVFP4QuantizeSwizzledKernel",
    "NVFP4QuantizeTMAKernel",
    "nvfp4_quantize_cute_dsl",
    "_get_compiled_kernel_nvfp4",
    "_get_compiled_kernel_nvfp4_tma",
]
