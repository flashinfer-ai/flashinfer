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

Fused Add + RMSNorm CuTe DSL Kernels
====================================

Includes:
- FusedAddRMSNormKernel: Fused residual add + RMSNorm
- FusedAddRMSNormQuantKernel: Fused residual add + RMSNorm + FP8 quantization
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from ..utils import (
    FLOAT8_E4M3_MAX,
    COPY_BITS,
    rcp_approx_ftz,
    cvt_and_store_f32_to_e4m3,
    get_ptr_as_int64,
    row_reduce_sum,
    predicate_k,
    compute_optimal_vec_size,
    compute_threads_per_row,
    make_tv_layout,
    _torch_dtype_to_str,
    get_cutlass_dtype,
)


# =============================================================================
# FusedAddRMSNormKernel
# =============================================================================


class FusedAddRMSNormKernel:
    """
    Fused Residual Add + RMSNorm Kernel using CuTe-DSL.

    Computes:
    1. residual = input + residual (in-place update)
    2. input = residual / sqrt(mean(residual^2) + eps) * (weight + weight_bias)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 0.0,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias

        # Vectorization parameters: use optimal vec_size for warp utilization
        elem_bits = dtype.width
        max_vec_size = COPY_BITS // elem_bits
        self.vec_size = compute_optimal_vec_size(H, max_vec_size)
        self.copy_bits = self.vec_size * elem_bits

        self.threads_per_row = compute_threads_per_row(H, self.vec_size)
        self.num_threads = self.threads_per_row
        self.num_warps = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1, (H // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

    def _smem_size_in_bytes(self) -> int:
        # Only reduction buffer needed (register-based approach)
        return self.num_warps * 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (1, self.cols_per_tile)

        self.kernel(mX, mR, mW, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        weight_bias = self.weight_bias
        threads_per_row = tv_layout.shape[0][0]
        num_warps = self.num_warps
        copy_bits = self.copy_bits

        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((num_warps,)),
            byte_alignment=4,
        )

        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gR = cute.local_tile(mR, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        mW_2d = cute.prepend_ones(mW, up_to_rank=2)

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn)
        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXgR = thr_copy.partition_S(gR)
        tXgW = thr_copy.partition_S(mW_2d)
        tXcX = thr_copy.partition_S(cX)
        tYgX = thr_copy.partition_D(gX)
        tYgR = thr_copy.partition_D(gR)

        # Register fragments - initialize to zero for proper handling of out-of-bounds threads
        tXrX = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
        tXrR = cute.make_rmem_tensor(tXgR.shape, mR.element_type)
        tXrW = cute.make_rmem_tensor(tXgW.shape, mW.element_type)
        tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
        tXrR.store(cute.zeros_like(tXrR, dtype=mR.element_type))
        tXrW.store(cute.zeros_like(tXrW, dtype=mW.element_type))

        tXpX = predicate_k(tXcX, limit=H)

        # Phase 1: Load input and residual from global to register
        cute.copy(copy_atom, tXgX, tXrX, pred=tXpX)
        cute.copy(copy_atom, tXgR, tXrR, pred=tXpX)

        x_in = tXrX.load().to(Float32)
        r_in = tXrR.load().to(Float32)
        x = x_in + r_in

        # Phase 2: Store x to residual (global)
        tXrR_out = x.to(mR.element_type)
        tXrR_store = cute.make_rmem_tensor(tYgR.shape, mR.element_type)
        tXrR_store.store(tXrR_out)

        cute.copy(copy_atom, tXrR_store, tYgR, pred=tXpX)

        # Phase 3: Compute sum of squares (x is kept in registers)
        x_sq = x * x
        sum_sq = row_reduce_sum(x_sq, threads_per_row, reduction_buffer)

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # Phase 4: Load weight from global to register
        cute.copy(copy_atom, tXgW, tXrW, pred=tXpX)

        w = tXrW.load().to(Float32)

        # output = x * rstd * (weight + weight_bias)
        # x is still in registers from Phase 1
        y = x * rstd * (w + Float32(weight_bias))

        tYrY = y.to(mX.element_type)
        tXrY = cute.make_rmem_tensor(tYgX.shape, mX.element_type)
        tXrY.store(tYrY)

        cute.copy(copy_atom, tXrY, tYgX, pred=tXpX)

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# FusedAddRMSNormQuantKernel
# =============================================================================


class FusedAddRMSNormQuantKernel:
    """
    Fused Residual Add + RMSNorm + FP8 Quantization Kernel.

    Computes:
    1. residual = input + residual (in-place update)
    2. output = clamp(residual / sqrt(mean(residual^2) + eps) * weight / scale, -448, 448)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 0.0,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias

        # Vectorization parameters: use optimal vec_size for warp utilization
        elem_bits = dtype.width
        max_vec_size = COPY_BITS // elem_bits
        self.vec_size = compute_optimal_vec_size(H, max_vec_size)
        self.copy_bits = self.vec_size * elem_bits

        self.threads_per_row = compute_threads_per_row(H, self.vec_size)
        self.num_threads = self.threads_per_row
        self.num_warps = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1, (H // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

    def _smem_size_in_bytes(self) -> int:
        # Only reduction buffer needed (register-based approach)
        return self.num_warps * 4

    @cute.jit
    def __call__(
        self,
        mY: cute.Tensor,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int32,
        scale: Float32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (1, self.cols_per_tile)

        self.kernel(
            mY, mX, mR, mW, M, scale, eps, enable_pdl, tv_layout, tiler_mn
        ).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mY: cute.Tensor,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int32,
        scale: Float32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        weight_bias = self.weight_bias
        threads_per_row = tv_layout.shape[0][0]
        num_warps = self.num_warps
        copy_bits = self.copy_bits
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks

        inv_scale = rcp_approx_ftz(scale)

        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((num_warps,)),
            byte_alignment=4,
        )

        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gR = cute.local_tile(mR, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        mW_2d = cute.prepend_ones(mW, up_to_rank=2)

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        tXgX = thr_copy_load.partition_S(gX)
        tXgR = thr_copy_load.partition_S(gR)
        tXgW = thr_copy_load.partition_S(mW_2d)
        tXcX = thr_copy_load.partition_S(cX)
        tYgR = thr_copy_load.partition_D(gR)

        # Register fragments - initialize to zero for proper handling of out-of-bounds threads
        tXrX = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
        tXrR = cute.make_rmem_tensor(tXgR.shape, mR.element_type)
        tXrW = cute.make_rmem_tensor(tXgW.shape, mW.element_type)
        tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
        tXrR.store(cute.zeros_like(tXrR, dtype=mR.element_type))
        tXrW.store(cute.zeros_like(tXrW, dtype=mW.element_type))

        tXpX = predicate_k(tXcX, limit=H)

        # Phase 1: Load input and residual from global to register
        cute.copy(copy_atom_load, tXgX, tXrX, pred=tXpX)
        cute.copy(copy_atom_load, tXgR, tXrR, pred=tXpX)

        x_in = tXrX.load().to(Float32)
        r_in = tXrR.load().to(Float32)
        x = x_in + r_in

        # Store x to residual (global)
        tXrR_out = x.to(mR.element_type)
        tXrR_store = cute.make_rmem_tensor(tYgR.shape, mR.element_type)
        tXrR_store.store(tXrR_out)
        cute.copy(copy_atom_load, tXrR_store, tYgR, pred=tXpX)

        # Phase 2: Compute sum of squares (x is kept in registers)
        x_sq = x * x
        sum_sq = row_reduce_sum(x_sq, threads_per_row, reduction_buffer)

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # Phase 3: Load weight from global to register
        cute.copy(copy_atom_load, tXgW, tXrW, pred=tXpX)
        w = tXrW.load().to(Float32)

        # output = x * rstd * (weight + weight_bias) * inv_scale
        # x is still in registers from Phase 1
        y = x * rstd * (w + Float32(weight_bias)) * inv_scale

        # Phase 4: Clamp and store to FP8 output using PTX scalar stores
        # (CuTe FP8 conversion requires vectorized ops, so we use PTX for scalar stores)
        # Store y to register tensor for element-wise access
        tYrY_f32 = cute.make_rmem_tensor(tXgX.shape, Float32)
        tYrY_f32.store(y)

        col_offset = tidx * vec_size
        for v in cutlass.range_constexpr(num_vec_blocks):
            for e in cutlass.range_constexpr(vec_size):
                idx = col_offset + v * threads_per_row * vec_size + e
                if idx < H:
                    # Clamp and convert - use flat index for register tensor
                    flat_idx = v * vec_size + e
                    clamped = max(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                    clamped = min(clamped, Float32(FLOAT8_E4M3_MAX))
                    # Use PTX to convert and store FP8 byte
                    out_offset = bidx * H + idx
                    out_ptr = get_ptr_as_int64(mY, Int32(out_offset))
                    cvt_and_store_f32_to_e4m3(clamped, out_ptr)

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# Compiled Kernel Getters
# =============================================================================


@functools.cache
def _get_compiled_fused_add_rmsnorm_kernel(
    dtype_str: str, H: int, weight_bias: float, enable_pdl: bool
):
    """Get a compiled Fused Add + RMSNorm kernel using TVM-FFI."""
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = FusedAddRMSNormKernel(dtype, H, weight_bias)

    sym_m = cute.sym_int()
    sym_row_stride_x = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_row_stride_r = cute.sym_int(divisibility=kernel_obj.vec_size)

    x_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
    )
    r_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_r, 1), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        r_fake,
        w_fake,
        Int32(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


@functools.cache
def _get_compiled_fused_add_rmsnorm_quant_kernel(
    dtype_str: str, out_dtype_str: str, H: int, weight_bias: float, enable_pdl: bool
):
    """Get a compiled Fused Add + RMSNorm + Quant kernel using TVM-FFI."""
    dtype = get_cutlass_dtype(dtype_str)
    out_dtype = get_cutlass_dtype(out_dtype_str)
    kernel_obj = FusedAddRMSNormQuantKernel(dtype, H, weight_bias)

    sym_m = cute.sym_int()
    sym_row_stride_y = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_row_stride_x = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_row_stride_r = cute.sym_int(divisibility=kernel_obj.vec_size)

    y_fake = cute.runtime.make_fake_tensor(
        out_dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
    )
    x_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
    )
    r_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_r, 1), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        y_fake,
        x_fake,
        r_fake,
        w_fake,
        Int32(1),
        Float32(1.0),  # scale
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


# =============================================================================
# CuTe DSL API Functions
# =============================================================================


def fused_add_rmsnorm_cute(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL Fused Add + RMSNorm implementation.

    Supports arbitrary stride - no need to call contiguous().
    Last dimension must be contiguous (stride[-1] == 1).
    """

    H = input.shape[-1]
    M = input.shape[0]

    dtype_str = _torch_dtype_to_str(input.dtype)
    kernel = _get_compiled_fused_add_rmsnorm_kernel(
        dtype_str, H, weight_bias, enable_pdl
    )
    kernel(input, residual, weight, M, eps)


def fused_add_rmsnorm_quant_cute(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL Fused Add + RMSNorm + FP8 quantization implementation.

    Supports arbitrary stride - no need to call contiguous().
    Last dimension must be contiguous (stride[-1] == 1).
    """

    H = input.shape[-1]
    M = input.shape[0]

    dtype_str = _torch_dtype_to_str(input.dtype)
    out_dtype_str = _torch_dtype_to_str(out.dtype)
    kernel = _get_compiled_fused_add_rmsnorm_quant_kernel(
        dtype_str, out_dtype_str, H, weight_bias, enable_pdl
    )
    kernel(
        out,
        input,
        residual,
        weight,
        M,
        scale,
        eps,
    )


__all__ = [
    # Kernel classes
    "FusedAddRMSNormKernel",
    "FusedAddRMSNormQuantKernel",
    # Compiled kernel getters
    "_get_compiled_fused_add_rmsnorm_kernel",
    "_get_compiled_fused_add_rmsnorm_quant_kernel",
    # CuTe DSL APIs
    "fused_add_rmsnorm_cute",
    "fused_add_rmsnorm_quant_cute",
]
