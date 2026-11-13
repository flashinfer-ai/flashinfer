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

LayerNorm CuTe DSL Kernels
==========================

Includes:
- LayerNormKernel: Traditional LayerNorm with mean and variance normalization
- LayerNormQuantKernel: LayerNorm + FP8 quantization
"""

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from ..utils import (
    COPY_BITS,
    get_fp8_max,
    rcp_approx_ftz,
    cvt_and_store_f32_to_fp8_hw,
    cvt_and_store_f32_to_fp8_sw,
    cvt_and_store_8xf32_to_fp8_hw,
    cvt_and_store_4xf32_to_fp8_hw,
    cvt_and_store_2xf32_to_fp8_hw,
    has_hw_fp8_cvt,
    get_ptr_as_int64,
    get_sm_version,
    row_reduce_sum,
    row_reduce_sum_multirow,
    predicate_k,
    compute_optimal_vec_size,
    compute_threads_per_row,
    make_tv_layout,
    _torch_dtype_to_str,
    get_cutlass_dtype,
)
from .rmsnorm import RMSNormKernel


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
        M: Int64,
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
        M: Int64,
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
# LayerNormQuantKernel
# =============================================================================


class LayerNormQuantKernel:
    """
    LayerNorm + FP8 Quantization Kernel using CuTe-DSL.

    Computes: output = ((input - mean) / sqrt(variance + eps) * gamma + beta) / scale,
    clamped to the finite range of the FP8 output dtype (E4M3 or E5M2).

    Derived from RMSNormQuantKernel (multi-row tiles, cluster reduction,
    cp.async staging, vectorized FP8 stores). gamma/beta are float32, so a
    dedicated tiled copy loads them for full tiles; tail tiles fall back to
    predicated per-element reads.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        use_hw_fp8: bool = True,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.use_hw_fp8 = use_hw_fp8
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        self.cluster_n = RMSNormKernel._compute_cluster_n(H, dtype, self.sm_version)
        self.H_per_cta = H // self.cluster_n

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = self.H_per_cta & (-self.H_per_cta)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = RMSNormKernel._compute_threads_per_row(self.H_per_cta)
        self.num_threads = RMSNormKernel._compute_num_threads(self.H_per_cta)
        # Widening to 256 threads at H_per_cta == 8192 helps only on SM120
        # (RTX 5090: 0.92 -> 1.23 TB/s at batch 1024); B200 regresses 6-17%.
        if self.num_threads < 256 and (
            self.H_per_cta > 8192 or (self.H_per_cta == 8192 and self.sm_version == 120)
        ):
            self.num_threads = 256
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        if self.copy_bits >= 32:
            tile_bytes = self.rows_per_block * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = tile_bytes <= props.shared_memory_per_block_optin // 2
        else:
            self.use_async_copy = False

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = (
                self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            )
        else:
            tile_bytes = 0

        if self.cluster_n == 1:
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            reduction_bytes = (
                self.rows_per_block * self.warps_per_row * self.cluster_n * 4
            )

        mbar_bytes = 8 if self.cluster_n > 1 else 0
        return tile_bytes + reduction_bytes + mbar_bytes

    @cute.jit
    def __call__(
        self,
        mY: cute.Tensor,
        mX: cute.Tensor,
        mGamma: cute.Tensor,
        mBeta: cute.Tensor,
        M: Int64,
        mS: cute.Tensor,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        cluster_n = self.cluster_n

        self.kernel(
            mY, mX, mGamma, mBeta, M, mS, eps, enable_pdl, tv_layout, tiler_mn
        ).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
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
        M: Int64,
        mS: cute.Tensor,
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
        cluster_n = self.cluster_n
        cols_per_tile = self.cols_per_tile
        copy_bits = self.copy_bits
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
        num_elems = vec_size * num_vec_blocks
        # Columns covered by the tile grid beyond H; they carry stale smem
        # data on the cp.async path and would bias mean/variance.
        has_tail = cols_per_tile * cluster_n != H

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        inv_scale = rcp_approx_ftz(mS[0])

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row
        col_offset = lane_in_row * vec_size

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + compute mean =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

        x = tXrX.load().to(Float32)

        # cp.async does not zero-fill predicated-off tail columns; they hold
        # whatever a previous tile left in smem and must not enter the mean.
        if cutlass.const_expr(has_tail and self.use_async_copy):
            x_reg = cute.make_rmem_tensor(x.shape, Float32)
            x_reg.store(x)
            for mi in cutlass.range_constexpr(num_elems):
                m_col = (
                    cluster_y * cols_per_tile
                    + col_offset
                    + (mi // vec_size) * threads_per_row * vec_size
                    + (mi % vec_size)
                )
                if m_col >= H:
                    x_reg[mi] = Float32(0.0)
            xm = x_reg.load()
        else:
            xm = x

        sum_x = row_reduce_sum_multirow(
            xm, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )
        mean = sum_x / Float32(H)

        # The second reduction reuses the same buffer and mbarrier. Block
        # path: keep round-2 writes from racing round-1 reads. Cluster path:
        # every CTA must complete mbarrier phase 0 before any CTA issues
        # round-2 stores, or those bytes would count toward phase 0.
        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Compute variance = E[(x - mean)^2] =====
        # Tail columns have x=0 (or stale data), so diff = -mean would leak
        # mean^2 into the variance; zero them before the reduction.
        if cutlass.const_expr(has_tail):
            diff_sq_reg = cute.make_rmem_tensor(x.shape, Float32)
            diff_sq_reg.store((xm - mean) * (xm - mean))
            for di in cutlass.range_constexpr(num_elems):
                d_col = (
                    cluster_y * cols_per_tile
                    + col_offset
                    + (di // vec_size) * threads_per_row * vec_size
                    + (di % vec_size)
                )
                if d_col >= H:
                    diff_sq_reg[di] = Float32(0.0)
            diff_sq = diff_sq_reg.load()
        else:
            diff_sq = (xm - mean) * (xm - mean)

        sum_diff_sq = row_reduce_sum_multirow(
            diff_sq,
            threads_per_row,
            reduction_buffer,
            mbar_ptr,
            cluster_n,
            phase=1,
        )
        variance = sum_diff_sq / Float32(H)
        rstd = cute.math.rsqrt(variance + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize, quantize, and store FP8 output =====
        # Re-load x from shared memory to relieve register pressure.
        # Without this, x (up to 128 FP32 values/thread at large H) must
        # survive across the reductions + barriers, causing spills to local
        # mem.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        # gamma/beta are float32 while the input tile is fp16/bf16, so the
        # input tiled copy cannot be reused. Full tiles use a dedicated tiled
        # copy with the same thread mapping and the value mode split into
        # (w_vec, vec_size // w_vec) 128-bit pieces, which keeps the fragment
        # flat order identical to the input fragment. Scalar predicated loads
        # would serialize on L2 latency and dominate the kernel time.
        gamma_reg = cute.make_rmem_tensor(x.shape, Float32)
        beta_reg = cute.make_rmem_tensor(x.shape, Float32)
        if cutlass.const_expr(has_tail):
            gamma_reg.store(cute.zeros_like(gamma_reg, dtype=Float32))
            beta_reg.store(cute.zeros_like(beta_reg, dtype=Float32))
            for gi in cutlass.range_constexpr(num_elems):
                g_col = (
                    cluster_y * cols_per_tile
                    + col_offset
                    + (gi // vec_size) * threads_per_row * vec_size
                    + (gi % vec_size)
                )
                if g_col < H:
                    gamma_reg[gi] = mGamma[g_col]
                    beta_reg[gi] = mBeta[g_col]
        else:
            w_vec = min(vec_size, 4)
            w_halves = vec_size // w_vec
            tv_layout_w = cute.make_layout(
                (
                    (threads_per_row, rows_per_block),
                    ((w_vec, w_halves), num_vec_blocks),
                ),
                stride=(
                    (vec_size * rows_per_block, 1),
                    (
                        (rows_per_block, w_vec * rows_per_block),
                        rows_per_block * vec_size * threads_per_row,
                    ),
                ),
            )
            copy_atom_w = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mGamma.element_type,
                num_bits_per_copy=w_vec * 32,
            )
            tiled_copy_W = cute.make_tiled_copy(copy_atom_w, tv_layout_w, tiler_mn)
            thr_copy_W = tiled_copy_W.get_slice(tidx)

            mGamma_2d = cute.make_tensor(
                mGamma.iterator,
                cute.prepend(
                    mGamma.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
                ),
            )
            mBeta_2d = cute.make_tensor(
                mBeta.iterator,
                cute.prepend(
                    mBeta.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
                ),
            )
            gG = cute.local_tile(mGamma_2d, tiler_mn, (0, cluster_y))
            gB = cute.local_tile(mBeta_2d, tiler_mn, (0, cluster_y))

            tWgG = thr_copy_W.partition_S(gG)
            tWgB = thr_copy_W.partition_S(gB)
            tWrG = cute.make_fragment_like(tWgG)
            tWrB = cute.make_fragment_like(tWgB)
            cute.copy(copy_atom_w, tWgG, tWrG)
            cute.copy(copy_atom_w, tWgB, tWrB)

            for wi in cutlass.range_constexpr(num_elems):
                gamma_reg[wi] = tWrG[wi]
                beta_reg[wi] = tWrB[wi]

        gamma = gamma_reg.load()
        beta = beta_reg.load()
        y = ((x - mean) * rstd * gamma + beta) * inv_scale

        tYrY_f32 = cute.make_rmem_tensor(tXrX.shape, Float32)
        tYrY_f32.store(y)

        # Compute actual_row in int64 so that, with M widened to Int64,
        # bidx * rows_per_block does not overflow int32 before being compared
        # against M or used in the address arithmetic below.
        actual_row = Int64(bidx) * rows_per_block + row_in_block
        fp8_max = get_fp8_max(mY.element_type)

        if cutlass.const_expr(self.use_hw_fp8 and vec_size == 8):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 8 <= H and actual_row < M:
                    base = v * 8
                    cvt_and_store_8xf32_to_fp8_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        tYrY_f32[base + 2],
                        tYrY_f32[base + 3],
                        tYrY_f32[base + 4],
                        tYrY_f32[base + 5],
                        tYrY_f32[base + 6],
                        tYrY_f32[base + 7],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                        mY.element_type,
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-fp8_max))
                            clamped = min(clamped, Float32(fp8_max))
                            cvt_and_store_f32_to_fp8_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int64(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                                mY.element_type,
                            )
        elif cutlass.const_expr(self.use_hw_fp8 and vec_size == 4):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 4 <= H and actual_row < M:
                    base = v * 4
                    cvt_and_store_4xf32_to_fp8_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        tYrY_f32[base + 2],
                        tYrY_f32[base + 3],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                        mY.element_type,
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-fp8_max))
                            clamped = min(clamped, Float32(fp8_max))
                            cvt_and_store_f32_to_fp8_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int64(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                                mY.element_type,
                            )
        elif cutlass.const_expr(self.use_hw_fp8 and vec_size == 2):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 2 <= H and actual_row < M:
                    base = v * 2
                    cvt_and_store_2xf32_to_fp8_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                        mY.element_type,
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-fp8_max))
                            clamped = min(clamped, Float32(fp8_max))
                            cvt_and_store_f32_to_fp8_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int64(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                                mY.element_type,
                            )
        else:
            for v in cutlass.range_constexpr(num_vec_blocks):
                for e in cutlass.range_constexpr(vec_size):
                    local_col = col_offset + v * threads_per_row * vec_size + e
                    abs_col = cluster_y * cols_per_tile + local_col
                    if abs_col < H and actual_row < M:
                        flat_idx = v * vec_size + e
                        clamped = max(tYrY_f32[flat_idx], Float32(-fp8_max))
                        clamped = min(clamped, Float32(fp8_max))
                        out_ptr = get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        )
                        if self.use_hw_fp8:
                            cvt_and_store_f32_to_fp8_hw(
                                clamped, out_ptr, mY.element_type
                            )
                        else:
                            cvt_and_store_f32_to_fp8_sw(
                                clamped, out_ptr, mY.element_type
                            )

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

    # 64-bit M and row strides so the offset arithmetic does not overflow
    # when M * H exceeds INT32_MAX.
    sym_m = cute.sym_int(64)
    sym_row_stride_y = cute.sym_int64(divisibility=kernel_obj.vec_size)
    sym_row_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)

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
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


@functools.cache
def _get_compiled_layernorm_quant_kernel(
    dtype_str: str,
    out_dtype_str: str,
    H: int,
    enable_pdl: bool,
    use_hw_fp8: bool = True,
    sm_version: int = 80,
    contiguous: bool = True,
):
    """Get a compiled LayerNorm + Quant kernel using TVM-FFI.

    See _get_compiled_rmsnorm_kernel for contiguous parameter semantics.
    gamma/beta are always float32, matching the CUDA backend contract.
    """
    dtype = get_cutlass_dtype(dtype_str)
    out_dtype = get_cutlass_dtype(out_dtype_str)
    kernel_obj = LayerNormQuantKernel(
        dtype, H, use_hw_fp8=use_hw_fp8, sm_version=sm_version
    )

    # 64-bit M so row-index arithmetic (row * H) does not overflow.
    sym_m = cute.sym_int(64)

    if contiguous:
        in_align = math.gcd(128, H * (dtype.width // 8))
        out_align = math.gcd(128, H * (out_dtype.width // 8))
        x_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=in_align
        )
        y_fake = cute.runtime.make_fake_compact_tensor(
            out_dtype, (sym_m, H), stride_order=(1, 0), assumed_align=out_align
        )
    else:
        sym_row_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_y = cute.sym_int64(divisibility=kernel_obj.vec_size)
        x_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
        )
        y_fake = cute.runtime.make_fake_tensor(
            out_dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
        )

    gamma_fake = cute.runtime.make_fake_compact_tensor(Float32, (H,), assumed_align=16)
    beta_fake = cute.runtime.make_fake_compact_tensor(Float32, (H,), assumed_align=16)
    s_fake = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=4)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        y_fake,
        x_fake,
        gamma_fake,
        beta_fake,
        Int64(1),
        s_fake,
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


def layernorm_quant_cute(
    out: torch.Tensor,
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL LayerNorm + FP8 quantization implementation.

    Expects contiguous 2D input/output (matching the CUDA backend contract).
    """
    shape = input.shape
    H = shape[-1]
    M = shape[0]

    # The contiguous compile path bakes the row stride in as a constexpr int
    # and computes row*H in int32; switch to symbolic int64 strides when the
    # element count exceeds INT32_MAX.
    contiguous = M * H <= 2**31 - 1

    dtype_str = _torch_dtype_to_str(input.dtype)
    out_dtype_str = _torch_dtype_to_str(out.dtype)
    kernel = _get_compiled_layernorm_quant_kernel(
        dtype_str,
        out_dtype_str,
        H,
        enable_pdl,
        use_hw_fp8=has_hw_fp8_cvt(input.device),
        sm_version=get_sm_version(input.device),
        contiguous=contiguous,
    )
    kernel(out, input, gamma, beta, M, scale, eps)


__all__ = [
    # Kernel classes
    "LayerNormKernel",
    "LayerNormQuantKernel",
    # Compiled kernel getters
    "_get_compiled_layernorm_kernel",
    "_get_compiled_layernorm_quant_kernel",
    # CuTe DSL APIs
    "layernorm_cute",
    "layernorm_quant_cute",
]
