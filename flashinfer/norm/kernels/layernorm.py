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

        # float32 gamma/beta have different vec_size (128-bit / 32 bits = 4 elements max)
        max_vec_size_f32 = COPY_BITS // 32  # = 4
        self.vec_size_f32 = compute_optimal_vec_size(H, max_vec_size_f32)
        self.copy_bits_f32 = self.vec_size_f32 * 32

        self.threads_per_row = compute_threads_per_row(H, self.vec_size)
        self.num_threads = self.threads_per_row
        self.num_warps = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1, (H // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        # For float32 gamma/beta vectorized load
        self.num_vec_blocks_f32 = max(
            1,
            (H // self.vec_size_f32 + self.threads_per_row - 1) // self.threads_per_row,
        )
        self.cols_per_tile_f32 = (
            self.vec_size_f32 * self.num_vec_blocks_f32 * self.threads_per_row
        )

    def _smem_size_in_bytes(self) -> int:
        # Shared memory for:
        # - gamma/beta f32 tiles: cols_per_tile_f32 * 4 * 2
        # - gamma/beta input dtype tiles: cols_per_tile * elem_bytes * 2
        # - reduction buffers: 2 * num_warps * 4
        elem_bytes = self.dtype.width // 8
        return (
            self.cols_per_tile_f32 * 4 * 2
            + self.cols_per_tile * elem_bytes * 2
            + 2 * self.num_warps * 4
        )

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

        # Layout for gamma/beta (float32)
        tv_shape_f32, tv_stride_f32 = make_tv_layout(
            self.threads_per_row,
            self.vec_size_f32,
            self.num_vec_blocks_f32,
        )
        tv_layout_f32 = cute.make_layout(tv_shape_f32, stride=tv_stride_f32)
        tiler_mn_f32 = (1, self.cols_per_tile_f32)

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
            tv_layout_f32,
            tiler_mn_f32,
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
        tv_layout_f32: cute.Layout,
        tiler_mn_f32: cute.Shape,
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
        copy_bits_f32 = self.copy_bits_f32

        smem = cutlass.utils.SmemAllocator()

        # Shared memory tiles for gamma, beta (float32)
        sGamma_f32 = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout(tiler_mn_f32, order=(1, 0)),
            byte_alignment=16,
        )
        sBeta_f32 = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout(tiler_mn_f32, order=(1, 0)),
            byte_alignment=16,
        )

        # Shared memory tiles for gamma, beta in input dtype (for matching shape with x)
        sGamma = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        sBeta = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

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

        # Expand gamma and beta to 2D for tiled copy (float32)
        mGamma_2d = cute.prepend_ones(mGamma, up_to_rank=2)
        mBeta_2d = cute.prepend_ones(mBeta, up_to_rank=2)

        # Identity tensor for gamma/beta bounds checking
        idGamma = cute.make_identity_tensor(mGamma_2d.shape)
        cGamma = cute.local_tile(idGamma, tiler_mn_f32, (0, 0))

        # Copy atom for input (input dtype) - sync load
        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        # Copy atom for gamma/beta (float32) - load to shared memory
        copy_atom_load_f32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Float32,
            num_bits_per_copy=copy_bits_f32,
        )

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
        tiled_copy_load_f32 = cute.make_tiled_copy(
            copy_atom_load_f32, tv_layout_f32, tiler_mn_f32
        )

        thr_copy_load = tiled_copy_load.get_slice(tidx)
        thr_copy_load_f32 = tiled_copy_load_f32.get_slice(tidx)

        # Partitions for input
        tXgX = thr_copy_load.partition_S(gX)
        tXgY = thr_copy_load.partition_D(gY)
        tXcX = thr_copy_load.partition_S(cX)

        # Partitions for gamma/beta (float32)
        tGgGamma = thr_copy_load_f32.partition_S(mGamma_2d)
        tGsGamma = thr_copy_load_f32.partition_D(sGamma_f32)
        tGgBeta = thr_copy_load_f32.partition_S(mBeta_2d)
        tGsBeta = thr_copy_load_f32.partition_D(sBeta_f32)
        tGcGamma = thr_copy_load_f32.partition_S(cGamma)

        # Partitions for gamma/beta (input dtype)
        thr_copy_load.partition_D(sGamma)
        thr_copy_load.partition_D(sBeta)

        # Register fragments - initialize to zero for proper handling of out-of-bounds threads
        tXrX = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
        tXrGamma = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
        tXrBeta = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
        tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
        tXrGamma.store(cute.zeros_like(tXrGamma, dtype=mX.element_type))
        tXrBeta.store(cute.zeros_like(tXrBeta, dtype=mX.element_type))

        tXpX = predicate_k(tXcX, limit=H)
        tGpGamma = predicate_k(tGcGamma, limit=H)

        # Phase 1: Load input from global to register
        cute.copy(copy_atom_load, tXgX, tXrX, pred=tXpX)

        # Phase 1b: Load gamma/beta global -> shared (float32)
        cute.copy(copy_atom_load_f32, tGgGamma, tGsGamma, pred=tGpGamma)
        cute.copy(copy_atom_load_f32, tGgBeta, tGsBeta, pred=tGpGamma)

        cute.arch.barrier()

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

        # Phase 3: Load gamma/beta directly from float32 shared memory
        # Avoid converting to bf16 and back to float32 which loses precision
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
                    gamma_reg[reg_idx] = sGamma_f32[0, idx]
                    beta_reg[reg_idx] = sBeta_f32[0, idx]

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

    def tensor_api(
        out: torch.Tensor,
        input: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        M: int,
        eps: float,
    ) -> None:
        compiled_kernel(
            out,
            input,
            gamma,
            beta,
            Int32(M),
            Float32(eps),
        )

    return tensor_api


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

    H = input.shape[-1]
    M = input.shape[0]

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
