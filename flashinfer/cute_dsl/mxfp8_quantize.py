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
"""

import functools
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint8

from ..api_logging import flashinfer_api
from .fp4_common import (
    ld_global_v4_u32,
    st_global_u64,
    get_ptr_as_int64,
    habs2,
    hmax2,
    bfloat2_habs2,
    bfloat2_hmax2,
)
from .quantization_utils import (
    # Constants
    SF_VEC_SIZE,
    INV_FLOAT8_E4M3_MAX,
    WARP_SIZE,
    ELTS_PER_THREAD,
    THREADS_PER_SF,
    SF_BLOCKS_PER_WARP,
    ROW_TILE_SIZE,
    # Intrinsics
    hmax_reduce_to_f32,
    bfloat2_hmax_reduce_to_f32,
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
    half2_to_fp8x2_scaled,
    bfloat2_to_fp8x2_scaled,
    pack_fp8x8_to_u64,
    reduce_max_4threads,
    compute_sf_index_swizzled_128x4_gpu,
)


# =============================================================================
# CuTe-DSL Kernel Class for Linear Layout
# =============================================================================


class MXFP8QuantizeLinearKernel:
    """
    MXFP8 quantization kernel optimized for LINEAR layout.
    Uses SF-block based iteration for efficient memory access.
    """

    WARPS_PER_BLOCK = 16  # 16 warps = 512 threads per block

    def __init__(
        self,
        dtype: cutlass.Numeric,
        M: int,
        K: int,
        enable_pdl: bool = False,
    ):
        self.dtype = dtype
        self.M = M
        self.K = K
        self.is_bfloat16 = (dtype == cutlass.BFloat16)
        self.enable_pdl = enable_pdl

        assert K % SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // SF_VEC_SIZE
        self.total_sf_blocks = M * self.num_sf_blocks_per_row
        self.scale_output_size = self.total_sf_blocks

    def _compute_grid_size(self) -> int:
        sf_blocks_per_tb = self.WARPS_PER_BLOCK * SF_BLOCKS_PER_WARP
        min_grid = (self.total_sf_blocks + sf_blocks_per_tb - 1) // sf_blocks_per_tb
        target_grid = 132 * 4
        return min(min_grid, target_grid)

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        stream,
    ):
        threads_per_block = self.WARPS_PER_BLOCK * WARP_SIZE
        num_blocks = self._compute_grid_size()

        self.kernel(mInput, mOutput, mScales).launch(
            grid=[num_blocks, 1, 1],
            block=[threads_per_block, 1, 1],
            max_number_threads=[512, 1, 1],
            min_blocks_per_mp=4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        warp_idx = tidx // WARP_SIZE
        lane_idx = tidx % WARP_SIZE

        sf_idx_in_warp = lane_idx // THREADS_PER_SF
        thread_in_sf = lane_idx % THREADS_PER_SF

        sf_blocks_per_tb = self.WARPS_PER_BLOCK * SF_BLOCKS_PER_WARP
        total_sf_blocks = self.total_sf_blocks
        num_sf_blocks_per_row = self.num_sf_blocks_per_row

        sf_idx_base = bidx * sf_blocks_per_tb + warp_idx * SF_BLOCKS_PER_WARP + sf_idx_in_warp

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

            if cutlass.const_expr(self.is_bfloat16):
                abs0 = bfloat2_habs2(v0)
                abs1 = bfloat2_habs2(v1)
                abs2 = bfloat2_habs2(v2)
                abs3 = bfloat2_habs2(v3)
                max01 = bfloat2_hmax2(abs0, abs1)
                max23 = bfloat2_hmax2(abs2, abs3)
                max0123 = bfloat2_hmax2(max01, max23)
                local_max = bfloat2_hmax_reduce_to_f32(max0123)
            else:
                abs0 = habs2(v0)
                abs1 = habs2(v1)
                abs2 = habs2(v2)
                abs3 = habs2(v3)
                max01 = hmax2(abs0, abs1)
                max23 = hmax2(abs2, abs3)
                max0123 = hmax2(max01, max23)
                local_max = hmax_reduce_to_f32(max0123)

            global_max = reduce_max_4threads(local_max)

            inv_e4m3_max = Float32(INV_FLOAT8_E4M3_MAX)
            normalized_max = global_max * inv_e4m3_max
            scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
            scale_ue8m0 = scale_ue8m0_u32.to(Uint8)

            inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)

            if cutlass.const_expr(self.is_bfloat16):
                fp8_01 = bfloat2_to_fp8x2_scaled(v0, inv_scale)
                fp8_23 = bfloat2_to_fp8x2_scaled(v1, inv_scale)
                fp8_45 = bfloat2_to_fp8x2_scaled(v2, inv_scale)
                fp8_67 = bfloat2_to_fp8x2_scaled(v3, inv_scale)
            else:
                fp8_01 = half2_to_fp8x2_scaled(v0, inv_scale)
                fp8_23 = half2_to_fp8x2_scaled(v1, inv_scale)
                fp8_45 = half2_to_fp8x2_scaled(v2, inv_scale)
                fp8_67 = half2_to_fp8x2_scaled(v3, inv_scale)

            fp8_packed = pack_fp8x8_to_u64(fp8_01, fp8_23, fp8_45, fp8_67)

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
    - Row-based iteration instead of SF-block iteration
    - Padding row fast path - only zero out scale factors
    - Column loop inside kernel for better locality
    """

    WARPS_PER_BLOCK = 16  # 16 warps = 512 threads per block

    def __init__(
        self,
        dtype: cutlass.Numeric,
        M: int,
        K: int,
        enable_pdl: bool = False,
    ):
        self.dtype = dtype
        self.M = M
        self.K = K
        self.is_bfloat16 = (dtype == cutlass.BFloat16)
        self.enable_pdl = enable_pdl

        assert K % SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // SF_VEC_SIZE
        self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4

        # For swizzled layout, pad M to multiple of 128
        self.padded_M = ((M + ROW_TILE_SIZE - 1) // ROW_TILE_SIZE) * ROW_TILE_SIZE
        self.scale_output_size = self.padded_M * self.padded_sf_cols

        # Number of column threads (each thread processes 8 elements = 1 SF block worth)
        self.num_col_units = self.num_sf_blocks_per_row

    def _compute_grid_size(self) -> int:
        """Grid size = number of padded rows (one block per row with grid-stride)."""
        target_blocks_per_sm = 4
        estimated_sm_count = 132
        target_grid = estimated_sm_count * target_blocks_per_sm
        return min(self.padded_M, target_grid)

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        stream,
    ):
        threads_per_block = self.WARPS_PER_BLOCK * WARP_SIZE
        num_blocks = self._compute_grid_size()

        self.kernel(mInput, mOutput, mScales).launch(
            grid=[num_blocks, 1, 1],
            block=[threads_per_block, 1, 1],
            max_number_threads=[512, 1, 1],
            min_blocks_per_mp=4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
    ):
        """
        Row-based kernel for swizzled layout.

        Each block processes one or more rows (grid-stride).
        Within each row, threads iterate over columns.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        M = self.M
        padded_M = self.padded_M
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        padded_sf_cols = self.padded_sf_cols

        # Thread organization within a row:
        # - Each "column unit" needs 4 threads (THREADS_PER_SF)
        # - tidx determines which column unit and which thread within unit
        col_unit_idx = tidx // THREADS_PER_SF
        thread_in_unit = tidx % THREADS_PER_SF

        threads_per_block = self.WARPS_PER_BLOCK * WARP_SIZE
        col_units_per_block = threads_per_block // THREADS_PER_SF  # 512/4 = 128

        # Grid-stride loop over rows
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
                # Normal path: process actual data row
                sf_col_idx = col_unit_idx
                while sf_col_idx < num_sf_blocks_per_row:
                    # Calculate element position
                    elem_idx = sf_col_idx * SF_VEC_SIZE + thread_in_unit * ELTS_PER_THREAD

                    row_input = mInput[row_idx, None]
                    input_ptr_i64 = get_ptr_as_int64(row_input, elem_idx)

                    # Load 8 FP16 values (128 bits)
                    v0, v1, v2, v3 = ld_global_v4_u32(input_ptr_i64)

                    # Compute local max using Half2 SIMD
                    if cutlass.const_expr(self.is_bfloat16):
                        abs0 = bfloat2_habs2(v0)
                        abs1 = bfloat2_habs2(v1)
                        abs2 = bfloat2_habs2(v2)
                        abs3 = bfloat2_habs2(v3)
                        max01 = bfloat2_hmax2(abs0, abs1)
                        max23 = bfloat2_hmax2(abs2, abs3)
                        max0123 = bfloat2_hmax2(max01, max23)
                        local_max = bfloat2_hmax_reduce_to_f32(max0123)
                    else:
                        abs0 = habs2(v0)
                        abs1 = habs2(v1)
                        abs2 = habs2(v2)
                        abs3 = habs2(v3)
                        max01 = hmax2(abs0, abs1)
                        max23 = hmax2(abs2, abs3)
                        max0123 = hmax2(max01, max23)
                        local_max = hmax_reduce_to_f32(max0123)

                    # 4-thread reduction for this SF block
                    global_max = reduce_max_4threads(local_max)

                    # Compute scale
                    inv_e4m3_max = Float32(INV_FLOAT8_E4M3_MAX)
                    normalized_max = global_max * inv_e4m3_max
                    scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
                    scale_ue8m0 = scale_ue8m0_u32.to(Uint8)

                    # Compute inverse scale
                    inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)

                    # Quantize
                    if cutlass.const_expr(self.is_bfloat16):
                        fp8_01 = bfloat2_to_fp8x2_scaled(v0, inv_scale)
                        fp8_23 = bfloat2_to_fp8x2_scaled(v1, inv_scale)
                        fp8_45 = bfloat2_to_fp8x2_scaled(v2, inv_scale)
                        fp8_67 = bfloat2_to_fp8x2_scaled(v3, inv_scale)
                    else:
                        fp8_01 = half2_to_fp8x2_scaled(v0, inv_scale)
                        fp8_23 = half2_to_fp8x2_scaled(v1, inv_scale)
                        fp8_45 = half2_to_fp8x2_scaled(v2, inv_scale)
                        fp8_67 = half2_to_fp8x2_scaled(v3, inv_scale)

                    fp8_packed = pack_fp8x8_to_u64(fp8_01, fp8_23, fp8_45, fp8_67)

                    # Store FP8 output
                    row_output = mOutput[row_idx, None]
                    output_ptr_i64 = get_ptr_as_int64(row_output, elem_idx)
                    st_global_u64(output_ptr_i64, fp8_packed)

                    # Thread 0 of the 4-thread group stores swizzled scale factor
                    if thread_in_unit == Int32(0):
                        sf_offset = compute_sf_index_swizzled_128x4_gpu(
                            row_idx, sf_col_idx, padded_sf_cols
                        )
                        mScales[sf_offset] = scale_ue8m0

                    # Move to next column unit
                    sf_col_idx = sf_col_idx + col_units_per_block

                # Handle padding columns if needed (zero out scale factors)
                sf_col_idx = num_sf_blocks_per_row + col_unit_idx
                while sf_col_idx < padded_sf_cols:
                    if thread_in_unit == Int32(0):
                        sf_offset = compute_sf_index_swizzled_128x4_gpu(
                            row_idx, sf_col_idx, padded_sf_cols
                        )
                        mScales[sf_offset] = Uint8(0)
                    sf_col_idx = sf_col_idx + col_units_per_block

            # Move to next row (grid-stride)
            row_idx = row_idx + grid_dim_x

        # PDL: Signal that dependent kernels can start early
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# PyTorch Integration with TVM-FFI
# =============================================================================


@functools.cache
def _get_compiled_kernel_linear(
    is_bfloat16: bool,
    M: int,
    K: int,
    enable_pdl: bool = False,
) -> Tuple[Callable, int]:
    """Get or compile LINEAR layout kernel with TVM-FFI."""
    cutlass_dtype = cutlass.BFloat16 if is_bfloat16 else cutlass.Float16
    kernel_obj = MXFP8QuantizeLinearKernel(cutlass_dtype, M, K, enable_pdl)

    # Use symbolic M for dynamic batch sizes
    sym_m = cute.sym_int()

    # Create fake tensors for compilation
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    # Scale output size for linear layout = M * (K // SF_VEC_SIZE)
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
        stream_fake,
        options="--enable-tvm-ffi",
    )

    scale_output_size = kernel_obj.scale_output_size

    def tensor_api(
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        scales_tensor: torch.Tensor,
    ) -> None:
        compiled_kernel(
            input_tensor,
            output_tensor,
            scales_tensor,
        )

    return tensor_api, scale_output_size


@functools.cache
def _get_compiled_kernel_swizzled(
    is_bfloat16: bool,
    M: int,
    K: int,
    enable_pdl: bool = False,
) -> Tuple[Callable, int]:
    """Get or compile SWIZZLED layout kernel with TVM-FFI."""
    cutlass_dtype = cutlass.BFloat16 if is_bfloat16 else cutlass.Float16
    kernel_obj = MXFP8QuantizeSwizzledKernel(cutlass_dtype, M, K, enable_pdl)

    # Use symbolic M for dynamic batch sizes
    sym_m = cute.sym_int()

    # Create fake tensors for compilation
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    # Scale output size for swizzled layout
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
        stream_fake,
        options="--enable-tvm-ffi",
    )

    scale_output_size = kernel_obj.scale_output_size

    def tensor_api(
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        scales_tensor: torch.Tensor,
    ) -> None:
        compiled_kernel(
            input_tensor,
            output_tensor,
            scales_tensor,
        )

    return tensor_api, scale_output_size


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

    orig_shape = input.shape
    if input.dim() > 2:
        m = input.numel() // input.shape[-1]
        k = input.shape[-1]
        input = input.reshape(m, k)
    else:
        m, k = input.shape

    assert k % SF_VEC_SIZE == 0, f"K ({k}) must be divisible by SF_VEC_SIZE={SF_VEC_SIZE}"

    padded_k = ((k + alignment - 1) // alignment) * alignment

    if padded_k > k:
        # Pad input with zeros - padding columns must be zero to produce zero FP8 output
        input_padded = torch.zeros(m, padded_k, dtype=input.dtype, device=input.device)
        input_padded[:, :k] = input
    else:
        input_padded = input.contiguous()

    is_bfloat16 = input.dtype == torch.bfloat16

    if is_sf_swizzled_layout:
        kernel_fn, scale_output_size = _get_compiled_kernel_swizzled(
            is_bfloat16, m, padded_k, enable_pdl
        )
    else:
        kernel_fn, scale_output_size = _get_compiled_kernel_linear(
            is_bfloat16, m, padded_k, enable_pdl
        )

    fp8_output = torch.empty(m, padded_k, dtype=torch.uint8, device=input.device)
    scale_output = torch.empty(scale_output_size, dtype=torch.uint8, device=input.device)

    kernel_fn(input_padded, fp8_output, scale_output)

    fp8_tensor = fp8_output.view(torch.float8_e4m3fn)

    return fp8_tensor, scale_output


__all__ = [
    "MXFP8QuantizeLinearKernel",
    "MXFP8QuantizeSwizzledKernel",
    "mxfp8_quantize_cute_dsl",
]
