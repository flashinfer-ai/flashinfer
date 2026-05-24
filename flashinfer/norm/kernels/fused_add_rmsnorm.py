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
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from ..utils import (
    FLOAT8_E4M3_MAX,
    COPY_BITS,
    rcp_approx_ftz,
    cvt_and_store_f32_to_e4m3_hw,
    cvt_and_store_f32_to_e4m3_sw,
    cvt_and_store_8xf32_to_e4m3_hw,
    cvt_and_store_4xf32_to_e4m3_hw,
    cvt_and_store_2xf32_to_e4m3_hw,
    has_hw_fp8_cvt,
    get_ptr_as_int64,
    get_sm_version,
    row_reduce_sum_multirow,
    predicate_k,
    _torch_dtype_to_str,
    get_cutlass_dtype,
)

from .rmsnorm import RMSNormKernel


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
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        self.cluster_n = self._compute_cluster_n(H, dtype, self.sm_version)
        self.H_per_cta = H // self.cluster_n

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = self.H_per_cta & (-self.H_per_cta)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = RMSNormKernel._compute_threads_per_row(self.H_per_cta)
        self.num_threads = RMSNormKernel._compute_num_threads(self.H_per_cta)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        if self.copy_bits >= 32:
            tile_bytes_2 = 2 * self.rows_per_block * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = (
                tile_bytes_2 <= props.shared_memory_per_block_optin // 2
            )
        else:
            self.use_async_copy = False

    @staticmethod
    def _compute_cluster_n(H: int, dtype: cutlass.Numeric, sm_version: int) -> int:
        """Compute optimal cluster size for fused-add kernel (2 shared tiles).

        Because fused-add needs 2 tiles (input + residual) in shared memory,
        we target smem <= max_smem // 2 so that at least 2 blocks can
        co-schedule per SM for good occupancy.  If no cluster_n achieves
        that, fall back to the first cluster_n that fits at all.
        """
        if sm_version < 90:
            return 1

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        max_smem_bytes = props.shared_memory_per_block_optin
        elem_size = dtype.width // 8
        occupancy_target = max_smem_bytes // 2

        best_fit = 1
        for cluster_n in [1, 2, 4, 8, 16]:
            if H % cluster_n != 0:
                continue
            smem_needed = FusedAddRMSNormKernel._estimate_smem_bytes(
                H, cluster_n, elem_size
            )
            if smem_needed <= occupancy_target:
                return cluster_n
            if smem_needed <= max_smem_bytes and best_fit == 1:
                best_fit = cluster_n

        return best_fit

    @staticmethod
    def _estimate_smem_bytes(H: int, cluster_n: int, elem_size: int) -> int:
        """Estimate shared memory bytes (2 tiles for input + residual)."""
        H_per_cta = H // cluster_n
        threads_per_row = RMSNormKernel._compute_threads_per_row(H_per_cta)
        num_threads = RMSNormKernel._compute_num_threads(H_per_cta)
        rows_per_block = num_threads // threads_per_row
        warps_per_row = max(threads_per_row // 32, 1)

        max_vec_size = COPY_BITS // 8 // elem_size
        h_align = H_per_cta & (-H_per_cta)
        vec_size = min(h_align, max_vec_size)
        num_vec_blocks = max(
            1, (H_per_cta // vec_size + threads_per_row - 1) // threads_per_row
        )
        cols_per_tile = vec_size * num_vec_blocks * threads_per_row

        tile_bytes = 2 * rows_per_block * cols_per_tile * elem_size

        if cluster_n == 1:
            return tile_bytes + rows_per_block * warps_per_row * 4
        else:
            return (
                tile_bytes
                + rows_per_block * warps_per_row * cluster_n * 4
                + 8  # mbarrier
            )

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = (
                2 * self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
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
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int32,
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

        self.kernel(mX, mR, mW, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
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
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
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
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
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

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        # Partition residual (same load tiled copy)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output destinations
        tXgO = thr_copy_O.partition_D(gX)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + residual, compute h, reduce =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        # Write h to residual (global)
        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mX.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

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
        use_hw_fp8: bool = True,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias
        self.use_hw_fp8 = use_hw_fp8
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        self.cluster_n = FusedAddRMSNormKernel._compute_cluster_n(
            H, dtype, self.sm_version
        )
        self.H_per_cta = H // self.cluster_n

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = self.H_per_cta & (-self.H_per_cta)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = RMSNormKernel._compute_threads_per_row(self.H_per_cta)
        self.num_threads = RMSNormKernel._compute_num_threads(self.H_per_cta)
        if self.H_per_cta > 8192 and self.num_threads < 256:
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
            tile_bytes_2 = 2 * self.rows_per_block * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = (
                tile_bytes_2 <= props.shared_memory_per_block_optin // 2
            )
        else:
            self.use_async_copy = False

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = (
                2 * self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
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
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int32,
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

        self.kernel(mY, mX, mR, mW, M, mS, eps, enable_pdl, tv_layout, tiler_mn).launch(
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
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int32,
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
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        inv_scale = rcp_approx_ftz(mS[0])

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
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
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
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

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        # Partition residual (same load tiled copy)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition residual store destination (match non-quant kernel pattern)
        tRgO = thr_copy_O.partition_D(gR)
        tXgO_r = thr_copy_O.partition_D(gX)
        tRrO = cute.make_fragment_like(tXgO_r)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + residual, compute h, reduce =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        # Write h to residual (global)
        tRrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tRrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize, quantize, and store FP8 output =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias)) * inv_scale

        tYrY_f32 = cute.make_rmem_tensor(tXrX.shape, Float32)
        tYrY_f32.store(y)

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row
        actual_row = bidx * rows_per_block + row_in_block
        col_offset = lane_in_row * vec_size

        if cutlass.const_expr(self.use_hw_fp8 and vec_size == 8):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 8 <= H and actual_row < M:
                    base = v * 8
                    cvt_and_store_8xf32_to_e4m3_hw(
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
                                (Int32(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                            clamped = min(clamped, Float32(FLOAT8_E4M3_MAX))
                            cvt_and_store_f32_to_e4m3_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int32(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                            )
        elif cutlass.const_expr(self.use_hw_fp8 and vec_size == 4):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 4 <= H and actual_row < M:
                    base = v * 4
                    cvt_and_store_4xf32_to_e4m3_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        tYrY_f32[base + 2],
                        tYrY_f32[base + 3],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int32(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                            clamped = min(clamped, Float32(FLOAT8_E4M3_MAX))
                            cvt_and_store_f32_to_e4m3_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int32(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                            )
        elif cutlass.const_expr(self.use_hw_fp8 and vec_size == 2):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 2 <= H and actual_row < M:
                    base = v * 2
                    cvt_and_store_2xf32_to_e4m3_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int32(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                            clamped = min(clamped, Float32(FLOAT8_E4M3_MAX))
                            cvt_and_store_f32_to_e4m3_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int32(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                            )
        else:
            for v in cutlass.range_constexpr(num_vec_blocks):
                for e in cutlass.range_constexpr(vec_size):
                    local_col = col_offset + v * threads_per_row * vec_size + e
                    abs_col = cluster_y * cols_per_tile + local_col
                    if abs_col < H and actual_row < M:
                        flat_idx = v * vec_size + e
                        clamped = max(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                        clamped = min(clamped, Float32(FLOAT8_E4M3_MAX))
                        out_ptr = get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int32(actual_row), Int32(abs_col)), mY.layout
                            ),
                        )
                        if self.use_hw_fp8:
                            cvt_and_store_f32_to_e4m3_hw(clamped, out_ptr)
                        else:
                            cvt_and_store_f32_to_e4m3_sw(clamped, out_ptr)

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# Compiled Kernel Getters
# =============================================================================


@functools.cache
def _get_compiled_fused_add_rmsnorm_kernel(
    dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
    contiguous: bool = True,
):
    """Get a compiled Fused Add + RMSNorm kernel using TVM-FFI.

    When contiguous=True, tensors are compiled with compact (dense) layouts for
    optimal codegen. When False, symbolic row strides are used to support
    arbitrary row strides at the cost of some performance.
    """
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = FusedAddRMSNormKernel(dtype, H, weight_bias, sm_version=sm_version)

    sym_m = cute.sym_int()

    if contiguous:
        elem_bytes = dtype.width // 8
        tensor_align = math.gcd(128, H * elem_bytes)
        x_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
        r_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
    else:
        sym_row_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_r = cute.sym_int64(divisibility=kernel_obj.vec_size)
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
    dtype_str: str,
    out_dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    use_hw_fp8: bool = True,
    sm_version: int = 80,
    contiguous: bool = True,
):
    """Get a compiled Fused Add + RMSNorm + Quant kernel using TVM-FFI.

    See _get_compiled_fused_add_rmsnorm_kernel for contiguous parameter semantics.
    """
    dtype = get_cutlass_dtype(dtype_str)
    out_dtype = get_cutlass_dtype(out_dtype_str)
    kernel_obj = FusedAddRMSNormQuantKernel(
        dtype, H, weight_bias, use_hw_fp8=use_hw_fp8, sm_version=sm_version
    )

    sym_m = cute.sym_int()

    if contiguous:
        in_align = math.gcd(128, H * (dtype.width // 8))
        out_align = math.gcd(128, H * (out_dtype.width // 8))
        y_fake = cute.runtime.make_fake_compact_tensor(
            out_dtype, (sym_m, H), stride_order=(1, 0), assumed_align=out_align
        )
        x_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=in_align
        )
        r_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=in_align
        )
    else:
        sym_row_stride_y = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_r = cute.sym_int64(divisibility=kernel_obj.vec_size)
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
    s_fake = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=4)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        y_fake,
        x_fake,
        r_fake,
        w_fake,
        Int32(1),
        s_fake,
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

    Supports non-contiguous tensors (stride[-1] must be 1). Uses an optimized
    compact kernel for contiguous inputs and a general strided kernel otherwise.
    Both input and residual are modified in-place.
    """

    shape = input.shape
    H = shape[-1]
    M = shape[0]

    is_contiguous = input.is_contiguous() and residual.is_contiguous()
    dtype_str = _torch_dtype_to_str(input.dtype)
    kernel = _get_compiled_fused_add_rmsnorm_kernel(
        dtype_str,
        H,
        weight_bias,
        enable_pdl,
        get_sm_version(input.device),
        contiguous=is_contiguous,
    )
    kernel(input, residual, weight, M, eps)


def fused_add_rmsnorm_quant_cute(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL Fused Add + RMSNorm + FP8 quantization implementation.

    Supports non-contiguous tensors (stride[-1] must be 1). Uses an optimized
    compact kernel for contiguous inputs and a general strided kernel otherwise.
    Residual is modified in-place with h = input + residual.
    """

    shape = input.shape
    H = shape[-1]
    M = shape[0]

    is_contiguous = (
        input.is_contiguous() and residual.is_contiguous() and out.is_contiguous()
    )
    dtype_str = _torch_dtype_to_str(input.dtype)
    out_dtype_str = _torch_dtype_to_str(out.dtype)
    kernel = _get_compiled_fused_add_rmsnorm_quant_kernel(
        dtype_str,
        out_dtype_str,
        H,
        weight_bias,
        enable_pdl,
        use_hw_fp8=has_hw_fp8_cvt(input.device),
        sm_version=get_sm_version(input.device),
        contiguous=is_contiguous,
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
