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
Supports swizzled (128x4) scale factor layout.

"""

import functools
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint8

from ...api_logging import flashinfer_api
from ...cute_dsl.fp4_common import get_ptr_as_int64, st_global_u64
from ..quantization_cute_dsl_utils import (
    # MXFP4 Constants
    MXFP4_SF_VEC_SIZE,
    ROW_TILE_SIZE,
    # Low-level intrinsics
    compute_sf_index_swizzled_128x4_gpu,
    # High-level helpers (MXFP4)
    process_mxfp4_block_half,
    process_mxfp4_block_bfloat,
)


# Blocks per SM for occupancy target
_BLOCKS_PER_SM = 4

# Maximum threads per block
_MAX_THREADS_PER_BLOCK = 1024

# Thread configuration bounds
_MIN_THREADS = 128  # Minimum for reasonable occupancy
_MAX_THREADS = 512  # Maximum to avoid register pressure
_DEFAULT_THREADS = 256  # Default thread count


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


def _compute_optimal_threads_for_k(K: int) -> int:
    """
    Compute optimal thread count for 100% thread utilization.

    For MXFP4, each thread processes one SF block (32 elements).
    threads_per_row = K / 32 = num_sf_blocks_per_row

    For 100% utilization when processing multiple rows:
        threads_per_block % threads_per_row == 0

    We prefer LARGER thread counts (up to _MAX_THREADS) for better occupancy,
    while maintaining 100% thread utilization.

    Args:
        K: Number of columns (must be divisible by 32)

    Returns:
        Optimal number of threads per block
    """
    threads_per_row = K // MXFP4_SF_VEC_SIZE  # K / 32

    # For 100% utilization: threads_per_block % threads_per_row == 0
    # threads_per_block must be a multiple of threads_per_row

    if threads_per_row >= _MAX_THREADS:
        # Large K: use max threads, will need column loop
        return _MAX_THREADS

    # threads_per_block should be a multiple of threads_per_row
    if threads_per_row <= _MAX_THREADS:
        # Find largest multiple of threads_per_row <= _MAX_THREADS
        threads = (_MAX_THREADS // threads_per_row) * threads_per_row
        if threads >= _MIN_THREADS:
            return threads
        # If largest multiple is below _MIN_THREADS, use the smallest valid one
        threads = threads_per_row
        while threads < _MIN_THREADS:
            threads += threads_per_row
        if threads <= _MAX_THREADS:
            return threads

    # Fallback to default
    return _DEFAULT_THREADS


def _compute_swizzled_layout_sf_size(
    total_row: int, total_column: int, row_size: int = 128
) -> int:
    """Compute size of swizzled scale factor buffer."""
    padded_row = (total_row + row_size - 1) // row_size * row_size
    padded_column = (total_column + 3) // 4 * 4
    return padded_row * padded_column


# =============================================================================
# CuTe-DSL Kernel Class for MXFP4 Swizzled Layout
# =============================================================================


class MXFP4QuantizeSwizzledKernel:
    """
    MXFP4 quantization kernel optimized for SWIZZLED layout.

    Key optimizations:
    - Multi-row processing: threads process multiple rows per block when K is small
    - Dynamic thread count based on K for 100% thread utilization
    - Row-based iteration with grid-stride loop
    - Padding row fast path - only zero out scale factors

    Thread utilization optimization:
    - For small K: Multiple rows processed per block iteration
    - For large K: Single row with column loop

    Each thread processes one SF block (32 elements):
    - UE8M0 scale factors (unsigned 8-bit exponent-only)
    - E2M1 output format (4-bit, 2 values per byte)

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
        self.target_grid = (
            target_grid if target_grid is not None else _get_target_grid()
        )

        assert K % MXFP4_SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = K // MXFP4_SF_VEC_SIZE
        self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4

        # Compute optimal thread count for 100% utilization
        self.num_threads = _compute_optimal_threads_for_k(K)

        # Multi-row processing constants (compile-time)
        # threads_per_row = num_sf_blocks_per_row (1 thread per SF block)
        self.threads_per_row = self.num_sf_blocks_per_row

        # Determine if we can process multiple rows or need column loop
        if self.threads_per_row <= self.num_threads:
            # Small K: multiple rows per block
            self.rows_per_block = self.num_threads // self.threads_per_row
            self.needs_col_loop = False
        else:
            # Large K: one row per block with column loop
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
        threads_per_block = self.num_threads
        rows_per_block = self.rows_per_block

        # Compute grid size at runtime
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
        MXFP4 quantization kernel with swizzled scale factor layout.

        Dual-path kernel with compile-time selection:
        - Small K path: Multi-row processing for improved thread utilization
        - Large K path: Single row with column loop

        Each thread processes one SF block (32 elements):
        1. Load 32 bf16/fp16 elements (4 x 128-bit loads)
        2. Compute max absolute value using SIMD reduction
        3. Compute UE8M0 scale: ceil(log2(max / 6.0)) + 127
        4. Swizzle scale factor to 128x4 layout
        5. Scale elements and convert to E2M1
        6. Store 16 bytes (32 FP4 values)

        Note: For MXFP4 (UE8M0 scale format), global scale is NOT used in
        the scale computation, unlike NVFP4 (E4M3 scale format). The UE8M0
        format directly captures the per-block dynamic range.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        padded_sf_cols = self.padded_sf_cols
        num_threads = self.num_threads
        rows_per_block = self.rows_per_block
        threads_per_row = self.threads_per_row

        if cutlass.const_expr(not self.needs_col_loop):
            # ===== SMALL K PATH: Multi-row processing =====
            # Each block processes rows_per_block rows simultaneously
            # Thread maps to: row_in_block = tidx // threads_per_row
            #                 sf_idx = tidx % threads_per_row
            row_in_block = tidx // threads_per_row
            sf_idx_in_row = tidx % threads_per_row

            # Grid-stride loop over row batches
            row_batch_idx = bidx
            total_row_batches = cute.ceil_div(padded_M, rows_per_block)

            while row_batch_idx < total_row_batches:
                base_row = row_batch_idx * rows_per_block
                row_idx = base_row + row_in_block

                if row_idx < padded_M:
                    is_padding_row = row_idx >= M

                    if is_padding_row:
                        # Fast path: padding row - only zero out scale factors
                        # Each participating thread zeros one SF at a time
                        local_sf_idx = sf_idx_in_row
                        while local_sf_idx < padded_sf_cols:
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, local_sf_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)
                            local_sf_idx = local_sf_idx + threads_per_row
                    else:
                        # Normal path: process actual data row
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

                            # Write swizzled scale factor
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

                        # Handle padding SF columns (columns beyond actual K)
                        padding_sf_start = num_sf_blocks_per_row + sf_idx_in_row
                        while padding_sf_start < padded_sf_cols:
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, padding_sf_start, padded_sf_cols
                            )
                            mScales[sf_offset] = Uint8(0)
                            padding_sf_start = padding_sf_start + threads_per_row

                row_batch_idx = row_batch_idx + grid_dim_x

        else:
            # ===== LARGE K PATH: Single row with column loop =====
            # Grid-stride loop over rows
            row_idx = bidx
            while row_idx < padded_M:
                is_padding_row = row_idx >= M

                # Initialize sf_idx before control flow to satisfy DSL type requirements
                sf_idx = Int32(tidx)

                if is_padding_row:
                    # Fast path: padding row - only zero out scale factors
                    while sf_idx < padded_sf_cols:
                        sf_offset = compute_sf_index_swizzled_128x4_gpu(
                            row_idx, sf_idx, padded_sf_cols
                        )
                        mScales[sf_offset] = Uint8(0)
                        sf_idx = sf_idx + num_threads
                else:
                    # Normal path: process actual data row with column loop
                    num_sf_iters = (
                        num_sf_blocks_per_row + num_threads - 1
                    ) // num_threads

                    for sf_iter in range(num_sf_iters):
                        local_sf_idx = sf_iter * num_threads + tidx

                        if local_sf_idx < num_sf_blocks_per_row:
                            elem_base = local_sf_idx * MXFP4_SF_VEC_SIZE
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

                            # Write swizzled scale factor
                            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                                row_idx, local_sf_idx, padded_sf_cols
                            )
                            mScales[sf_offset] = scale_ue8m0

                            # Store 16 bytes (32 FP4 values = 2 x st.global.u64)
                            row_output = mOutput[row_idx, None]
                            out_base = local_sf_idx * (MXFP4_SF_VEC_SIZE // 2)
                            out_ptr0 = get_ptr_as_int64(row_output, out_base)
                            out_ptr1 = get_ptr_as_int64(row_output, out_base + Int32(8))
                            st_global_u64(out_ptr0, packed64_0)
                            st_global_u64(out_ptr1, packed64_1)

                    # Handle padding SF columns (columns beyond actual K)
                    padding_sf_start = num_sf_blocks_per_row + tidx
                    while padding_sf_start < padded_sf_cols:
                        sf_offset = compute_sf_index_swizzled_128x4_gpu(
                            row_idx, padding_sf_start, padded_sf_cols
                        )
                        mScales[sf_offset] = Uint8(0)
                        padding_sf_start = padding_sf_start + num_threads

                row_idx = row_idx + grid_dim_x

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
    enable_pdl: bool = False,
    target_grid: int = None,
) -> Callable:
    """
    Get or compile MXFP4 kernel with TVM-FFI.

    Cached by (K, dtype, pdl, target_grid) - M-agnostic compilation.
    """
    cutlass_dtype = cutlass.BFloat16 if is_bfloat16 else cutlass.Float16
    kernel_obj = MXFP4QuantizeSwizzledKernel(
        cutlass_dtype, K, enable_pdl, target_grid=target_grid
    )

    # Use symbolic M for dynamic batch sizes
    sym_m = cute.sym_int()

    # Create fake tensors for compilation
    # Input: [M, K] in fp16/bf16
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, K), stride_order=(1, 0), assumed_align=16
    )
    # Output: [M, K/2] in uint8 (2 FP4 values per byte)
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, K // 2), stride_order=(1, 0), assumed_align=16
    )
    # Scales: 1D swizzled buffer
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
def mxfp4_quantize_cute_dsl(
    input: torch.Tensor,
    enable_pdl: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to MXFP4 format using CuTe-DSL kernel.

    This is a GPU implementation matching FlashInfer's mxfp4_quantize() behavior:
    - Global scale computed as (448 * 6) / max(|input|)
    - UE8M0 scale factors
    - E2M1 output format (4-bit, 2 values per byte)
    - Swizzled (128x4) scale factor layout

    The kernel is compiled once per (K, dtype, pdl) combination and handles
    varying M (batch size) at runtime without recompilation.

    Args:
        input: Input tensor of shape [M, K] with dtype fp16/bf16
        enable_pdl: Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability (SM >= 9.0).

    Returns:
        Tuple of:
            - fp4_tensor: Quantized tensor of shape [M, K/2] with dtype uint8
            - scale_tensor: Scale factors as uint8 tensor (swizzled layout)
    """
    from ...utils import device_support_pdl

    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"Input dtype must be float16 or bfloat16, got {input.dtype}"
    )
    assert input.is_cuda, "Input must be on CUDA device"

    # Auto-detect PDL support based on device capability
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

    # Compute device-specific target grid for kernel compilation
    target_grid = _get_target_grid(input.device)

    # Compute M-dependent values
    num_sf_blocks_per_row = k // MXFP4_SF_VEC_SIZE
    padded_m = ((m + ROW_TILE_SIZE - 1) // ROW_TILE_SIZE) * ROW_TILE_SIZE
    padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4
    scale_output_size = padded_m * padded_sf_cols

    # Get or compile kernel
    kernel_fn = _get_compiled_kernel_mxfp4(is_bfloat16, k, enable_pdl, target_grid)

    # Allocate outputs
    # Output: [M, K/2] uint8 (2 FP4 values per byte)
    fp4_output = torch.empty(m, k // 2, dtype=torch.uint8, device=input.device)
    scale_output = torch.empty(
        scale_output_size, dtype=torch.uint8, device=input.device
    )

    # Launch kernel
    kernel_fn(input, fp4_output, scale_output, m, padded_m)

    # Reshape scale output to match CUDA backend format: [padded_total, num_sf_per_row]
    scale_output = scale_output.reshape(-1, num_sf_blocks_per_row)

    return fp4_output, scale_output


__all__ = [
    "MXFP4QuantizeSwizzledKernel",
    "mxfp4_quantize_cute_dsl",
    "_get_compiled_kernel_mxfp4",
]
