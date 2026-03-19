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

    target_grid = get_num_sm(input.device) * _BLOCKS_PER_SM

    num_sf_blocks_per_row = k // NVFP4_SF_VEC_SIZE

    if sf_layout == SF_LAYOUT_LINEAR:
        row_tile_size = 1
        # NOTE: When adding a TMA-based kernel, padded_m must be rounded up to the
        # TMA tile row dimension (e.g. round_up(m, tma_tile_rows)) and scale_output
        # must be trimmed to m * num_sf_blocks_per_row before returning.
        # See PR f4d10d9 for the analogous CUDA fix.
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

    num_blocks = min((padded_m + rows_per_block - 1) // rows_per_block, target_grid)

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
    "nvfp4_quantize_cute_dsl",
    "_get_compiled_kernel_nvfp4",
]
