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

LayerNorm CuTe DSL Kernel
=========================

Traditional LayerNorm with mean and variance normalization.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from ..utils import (
    COPY_BITS,
    row_reduce_sum,
    predicate_k,
    compute_optimal_vec_size,
    compute_threads_per_row,
    make_tv_layout,
    _torch_dtype_to_str,
    get_cutlass_dtype,
)


# =============================================================================
# LayerNormKernel
# =============================================================================


class LayerNormKernel:
    """
    Layer Normalization Kernel using CuTe-DSL.

    Computes: output = (input - mean) / sqrt(variance + eps) * gamma + beta
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
    ):
        self.dtype = dtype
        self.H = H

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
        # Two reduction buffers (sum and variance), one float32 slot per warp each
        return 2 * self.num_warps * 4

    @cute.jit
    def __call__(
        self,
        mY: cute.Tensor,
        mX: cute.Tensor,
        mGamma: cute.Tensor,
        mBeta: cute.Tensor,
        M: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        # Layout for input (float16/bfloat16)
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (1, self.cols_per_tile)

        self.kernel(
            mY,
            mX,
            mGamma,
            mBeta,
            M,
            eps,
            enable_pdl,
            tv_layout,
            tiler_mn,
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
        mGamma: cute.Tensor,
        mBeta: cute.Tensor,
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
        threads_per_row = tv_layout.shape[0][0]
        num_warps = self.num_warps
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks
        copy_bits = self.copy_bits

        smem = cutlass.utils.SmemAllocator()

        # Two reduction buffers: one for sum, one for variance
        reduction_buffer_sum = smem.allocate_tensor(
            Float32,
            cute.make_layout((num_warps,)),
            byte_alignment=4,
        )

        reduction_buffer_var = smem.allocate_tensor(
            Float32,
            cute.make_layout((num_warps,)),
            byte_alignment=4,
        )

        idX = cute.make_identity_tensor(mX.shape)

        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # Copy atom for input (input dtype) - sync load
        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)

        thr_copy_load = tiled_copy_load.get_slice(tidx)

        # Partitions for input
        tXgX = thr_copy_load.partition_S(gX)
        tXgY = thr_copy_load.partition_D(gY)
        tXcX = thr_copy_load.partition_S(cX)

        # Register fragment - initialize to zero for proper handling of out-of-bounds threads
        tXrX = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
        tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))

        tXpX = predicate_k(tXcX, limit=H)

        # Phase 1: Load input from global to register
        cute.copy(copy_atom_load, tXgX, tXrX, pred=tXpX)

        x = tXrX.load().to(Float32)
        sum_x = row_reduce_sum(x, threads_per_row, reduction_buffer_sum)

        mean = sum_x / Float32(H)

        # Phase 2: Compute variance = E[(x - mean)^2]
        # For invalid threads (col >= H), x=0 so diff = -mean, which would incorrectly
        # contribute mean^2 to variance. We zero out these positions before reduction.
        diff = x - mean
        diff_sq = diff * diff

        num_elems = vec_size * num_vec_blocks
        diff_sq_reg = cute.make_rmem_tensor(diff_sq.shape, Float32)
        diff_sq_reg.store(diff_sq)

        # Zero out invalid positions so they don't contribute to variance
        for i in cutlass.range_constexpr(num_elems):
            vec_idx = i % vec_size
            block_idx = i // vec_size
            col = tidx * vec_size + vec_idx + block_idx * vec_size * threads_per_row
            if col >= H:
                diff_sq_reg[i] = Float32(0.0)

        diff_sq_masked = diff_sq_reg.load()
        sum_diff_sq = row_reduce_sum(
            diff_sq_masked, threads_per_row, reduction_buffer_var
        )

        variance = sum_diff_sq / Float32(H)
        rstd = cute.math.rsqrt(variance + eps, fastmath=True)

        cute.arch.barrier()

        # Phase 3: Load gamma/beta directly from global memory into registers.
        # Each thread owns a disjoint range of columns so there is no sharing
        # between threads — staging through shared memory is unnecessary.
        gamma_reg = cute.make_rmem_tensor(x.shape, Float32)
        beta_reg = cute.make_rmem_tensor(x.shape, Float32)
        gamma_reg.store(cute.zeros_like(gamma_reg, dtype=Float32))
        beta_reg.store(cute.zeros_like(beta_reg, dtype=Float32))

        col_offset = tidx * vec_size
        for v in cutlass.range_constexpr(num_vec_blocks):
            for e in cutlass.range_constexpr(vec_size):
                idx = col_offset + v * threads_per_row * vec_size + e
                reg_idx = v * vec_size + e
                if idx < H:
                    gamma_reg[reg_idx] = mGamma[idx]
                    beta_reg[reg_idx] = mBeta[idx]

        gamma = gamma_reg.load()
        beta = beta_reg.load()

        # output = (x - mean) * rstd * gamma + beta
        y = (x - mean) * rstd * gamma + beta

        tYrY = y.to(mY.element_type)
        tXrY = cute.make_rmem_tensor(tXgY.shape, mY.element_type)
        tXrY.store(tYrY)

        cute.copy(copy_atom_load, tXrY, tXgY, pred=tXpX)

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# Compiled Kernel Getter
# =============================================================================


@functools.cache
def _get_compiled_layernorm_kernel(
    dtype_str: str, gamma_dtype_str: str, H: int, enable_pdl: bool
):
    """Get a compiled LayerNorm kernel using TVM-FFI."""
    dtype = get_cutlass_dtype(dtype_str)
    gamma_dtype = get_cutlass_dtype(gamma_dtype_str)
    kernel_obj = LayerNormKernel(dtype, H)

    sym_m = cute.sym_int()
    sym_row_stride_y = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_row_stride_x = cute.sym_int(divisibility=kernel_obj.vec_size)

    y_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
    )
    x_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
    )
    gamma_fake = cute.runtime.make_fake_compact_tensor(
        gamma_dtype, (H,), assumed_align=16
    )
    beta_fake = cute.runtime.make_fake_compact_tensor(
        gamma_dtype, (H,), assumed_align=16
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        y_fake,
        x_fake,
        gamma_fake,
        beta_fake,
        Int32(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


# =============================================================================
# CuTe DSL API Function
# =============================================================================


def layernorm_cute(
    out: torch.Tensor,
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL LayerNorm implementation.

    Supports arbitrary stride - no need to call contiguous().
    Last dimension must be contiguous (stride[-1] == 1).
    """

    shape = input.shape
    H = shape[-1]
    M = shape[0]

    dtype_str = _torch_dtype_to_str(input.dtype)
    gamma_dtype_str = _torch_dtype_to_str(gamma.dtype)
    kernel = _get_compiled_layernorm_kernel(dtype_str, gamma_dtype_str, H, enable_pdl)
    kernel(out, input, gamma, beta, M, eps)


__all__ = [
    # Kernel class
    "LayerNormKernel",
    # Compiled kernel getter
    "_get_compiled_layernorm_kernel",
    # CuTe DSL API
    "layernorm_cute",
]
