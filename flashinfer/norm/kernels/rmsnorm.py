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

RMSNorm CuTe DSL Kernels
========================

Includes:
- RMSNormKernel: Basic RMSNorm (also handles Gemma variant with weight_bias=1.0)
- QKRMSNormKernel: RMSNorm for 3D tensors [batch, heads, head_dim]
- RMSNormQuantKernel: RMSNorm + FP8 quantization
"""

import functools
import math
import operator

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
    has_hw_fp8_cvt,
    get_ptr_as_int64,
    warp_reduce,
    row_reduce_sum,
    row_reduce_sum_multirow,
    predicate_k,
    compute_optimal_vec_size,
    compute_threads_per_row,
    make_tv_layout,
    _torch_dtype_to_str,
    get_cutlass_dtype,
    get_num_sm,
)


# =============================================================================
# RMSNormKernel
# =============================================================================


class RMSNormKernel:
    """
    RMSNorm Kernel using CuTe-DSL with multi-row block processing.

    Computes: output = input / sqrt(mean(input^2) + eps) * (weight + weight_bias)

    Architecture follows the Blackwell CUTLASS example:
    1. Multi-row blocks for better wave utilization at small batch sizes
    2. Async global→shared copy when H alignment permits (H divisible by vec_size)
    3. Graceful fallback to sync copy for arbitrary H (e.g. H=111)
    4. Two-pass access from shared memory (sum-of-squares then normalize)
    5. Weight loaded via separate sync copy, overlapped with async wait
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

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        # vec_size must divide H so row addresses stay aligned for vectorized loads.
        # Largest power-of-2 factor of H, capped at max_vec_size.
        h_align = H & (-H)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        # cp.async requires at least 32-bit (4 byte) transfers
        self.use_async_copy = self.copy_bits >= 32

        self.threads_per_row = self._compute_threads_per_row(H)
        self.num_threads = self._compute_num_threads(H)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (H // self.vec_size + self.threads_per_row - 1) // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

    @staticmethod
    def _compute_threads_per_row(H: int) -> int:
        if H <= 64:
            return 8
        elif H <= 128:
            return 16
        elif H <= 3072:
            return 32
        elif H <= 6144:
            return 64
        elif H <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(H: int) -> int:
        return 128 if H <= 16384 else 256

    @staticmethod
    def _make_tv_layout(threads_per_row, rows_per_block, vec_size, num_vec_blocks):
        """Create Thread-Value layout for multi-row coalesced vectorized access."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = (
                self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            )
        else:
            tile_bytes = 0
        reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        return tile_bytes + reduction_bytes

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        self.kernel(mX, mW, mY, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            # --- Async path: global → shared → register ---
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(
                copy_atom_async, tv_layout, tiler_mn
            )
        else:
            # --- Sync path: global → register directly ---
            tiled_copy_load = cute.make_tiled_copy(
                copy_atom_sync, tv_layout, tiler_mn
            )

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output
        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            # Load weight while waiting for async copy
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            # Sync: load input directly to registers
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

            # Load weight
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(x_sq, threads_per_row, reduction_buffer)

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        if cutlass.const_expr(self.use_async_copy):
            # Reload x from shared memory (fresh load after barrier)
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# QKRMSNormKernel
# =============================================================================


class QKRMSNormKernel:
    """
    QK RMSNorm Kernel using CuTe-DSL for 3D tensors [batch, heads, head_dim].

    Supports arbitrary stride - no need for contiguous tensors.
    Each warp processes one (batch, head) pair independently.
    Uses warp-only reduction (no cross-warp shared memory sync needed).

    Computes: output[b,h,:] = input[b,h,:] / sqrt(mean(input[b,h,:]^2) + eps) * (weight + weight_bias)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        head_dim: int,
        weight_bias: float = 0.0,
        num_warps: int = 4,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.weight_bias = weight_bias
        self.num_warps = num_warps

        # Vectorization: each warp (32 threads) processes head_dim elements
        elem_bits = dtype.width
        max_vec_size = COPY_BITS // elem_bits  # 8 for float16/bfloat16
        self.vec_size = compute_optimal_vec_size(head_dim, max_vec_size)
        self.copy_bits = self.vec_size * elem_bits

        # Threads per warp is always 32
        self.threads_per_warp = 32
        self.num_threads = self.threads_per_warp * num_warps

        # Number of vectorized blocks per warp
        self.num_vec_blocks = max(
            1,
            (head_dim // self.vec_size + self.threads_per_warp - 1)
            // self.threads_per_warp,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_warp

    def _smem_size_in_bytes(self) -> int:
        # No shared memory needed - warp-only reduction
        return 0

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        B: Int32,
        N: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        num_blocks: Int32,
        stream,
    ):
        """Launch the QKRMSNorm kernel.

        Args:
            mX: Input tensor of shape [B, N, H] with arbitrary stride.
            mW: Weight tensor of shape [H].
            mY: Output tensor of shape [B, N, H] with arbitrary stride.
            B: Batch size.
            N: Number of heads.
            eps: Epsilon for numerical stability.
            enable_pdl: Enable PDL for SM90+.
            num_blocks: Number of blocks to launch.
            stream: CUDA stream.
        """
        # Use 32 threads per warp for warp-level layout
        tv_shape, tv_stride = make_tv_layout(32, self.vec_size, self.num_vec_blocks)
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)

        self.kernel(mX, mW, mY, B, N, eps, enable_pdl, tv_layout).launch(
            grid=[num_blocks, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        B: Int32,
        N: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
    ):
        """Device kernel for QKRMSNorm with 3D tensor support and arbitrary stride."""
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        head_dim = self.head_dim
        weight_bias = self.weight_bias
        num_warps = self.num_warps
        copy_bits = self.copy_bits

        # Thread indexing within block
        lane_idx = tidx % 32
        warp_idx = tidx // 32

        # Total workers and jobs
        grid_dim_x, _, _ = cute.arch.grid_dim()
        num_workers = grid_dim_x * num_warps
        worker_idx = bidx * num_warps + warp_idx

        # Total number of rows
        M = B * N

        # Create copy atom for vectorized loads/stores
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        # Expand weight to 2D for consistent tiling: [1, H]
        mW_2d = cute.prepend_ones(mW, up_to_rank=2)

        # Create tiled copy for warp-level access (32 threads)
        tiler_2d = (1, self.cols_per_tile)
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_2d)
        thr_copy = tiled_copy.get_slice(lane_idx)

        # Create identity tensor matching tile shape for bounds checking
        id2d = cute.make_identity_tensor(tiler_2d)

        # Weight and predicate are the same for all rows - compute once
        tXgW = thr_copy.partition_S(mW_2d)
        tXcX = thr_copy.partition_S(id2d)
        tXpX = predicate_k(tXcX, limit=head_dim)

        # Load weight once (same for all rows)
        tXrW = cute.make_rmem_tensor(tXgW.shape, mW.element_type)
        tXrW.store(cute.zeros_like(tXrW, dtype=mW.element_type))
        cute.copy(copy_atom, tXgW, tXrW, pred=tXpX)
        w = tXrW.load().to(Float32)

        # Each warp processes multiple rows with grid-stride loop
        row_idx = worker_idx
        while row_idx < M:
            batch_idx = row_idx // N
            head_idx = row_idx % N

            # Get 3D tile and collapse first two dims (both size 1) to 2D for tiled_copy
            gX = cute.group_modes(
                cute.local_tile(
                    mX, (1, 1, self.cols_per_tile), (batch_idx, head_idx, 0)
                ),
                0,
                2,
            )
            gY = cute.group_modes(
                cute.local_tile(
                    mY, (1, 1, self.cols_per_tile), (batch_idx, head_idx, 0)
                ),
                0,
                2,
            )

            # Partition tensors for this thread
            tXgX = thr_copy.partition_S(gX)
            tXgY = thr_copy.partition_D(gY)

            # Register fragment for input - initialize to zero
            tXrX = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))

            # Phase 1: Load input and compute sum of squares
            cute.copy(copy_atom, tXgX, tXrX, pred=tXpX)

            x = tXrX.load().to(Float32)
            x_sq = x * x

            # Reduce within register tensor first
            local_sum = x_sq.reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            )

            # Warp reduction for sum_sq
            sum_sq = warp_reduce(local_sum, operator.add, width=32)

            # Compute rstd
            mean_sq = sum_sq / Float32(head_dim)
            rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

            # output = input * rstd * (weight + weight_bias)
            # w is already loaded outside the loop
            y = x * rstd * (w + Float32(weight_bias))

            # Store output
            tYrY = y.to(mY.element_type)
            tXrY = cute.make_rmem_tensor(tXgY.shape, mY.element_type)
            tXrY.store(tYrY)

            cute.copy(copy_atom, tXrY, tXgY, pred=tXpX)

            # Next row for this warp
            row_idx = row_idx + num_workers

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# RMSNormQuantKernel
# =============================================================================


class RMSNormQuantKernel:
    """
    RMSNorm + FP8 Quantization Kernel using CuTe-DSL.

    Computes: output = clamp(input / sqrt(mean(input^2) + eps) * weight / scale, -448, 448)
    Then quantizes to FP8 E4M3.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 0.0,
        use_hw_fp8: bool = True,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias
        self.use_hw_fp8 = use_hw_fp8

        # Vectorization parameters: use optimal vec_size for warp utilization
        elem_bits = dtype.width
        max_vec_size_in = COPY_BITS // elem_bits  # 8 for fp16/bf16
        self.vec_size = compute_optimal_vec_size(H, max_vec_size_in)
        self.copy_bits = self.vec_size * elem_bits

        # For FP8 output: minimum 16 bits = 2 FP8 elements
        # Use same vec_size to keep layouts aligned, but ensure copy_bits_out >= 16
        self.vec_size_out = self.vec_size
        self.copy_bits_out = max(16, self.vec_size * 8)

        self.threads_per_row = compute_threads_per_row(H, self.vec_size)
        self.num_threads = self.threads_per_row
        self.num_warps = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1, (H // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

    def _smem_size_in_bytes(self) -> int:
        # Only reduction buffer needed
        return self.num_warps * 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        mS: cute.Tensor,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (1, self.cols_per_tile)

        self.kernel(mX, mW, mY, M, mS, eps, enable_pdl, tv_layout, tiler_mn).launch(
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
        mW: cute.Tensor,
        mY: cute.Tensor,
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
        weight_bias = self.weight_bias
        threads_per_row = tv_layout.shape[0][0]
        num_warps = self.num_warps
        copy_bits = self.copy_bits
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks

        inv_scale = rcp_approx_ftz(mS[0])

        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            Float32, cute.make_layout((num_warps,)), byte_alignment=4
        )

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        mW_2d = cute.prepend_ones(mW, up_to_rank=2)

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits
        )

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        tXgX = thr_copy_load.partition_S(gX)
        tXgW = thr_copy_load.partition_S(mW_2d)
        tXcX = thr_copy_load.partition_S(cX)

        # Register fragments - initialize to zero for proper handling of out-of-bounds threads
        tXrX = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
        tXrW = cute.make_rmem_tensor(tXgW.shape, mW.element_type)
        tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
        tXrW.store(cute.zeros_like(tXrW, dtype=mW.element_type))

        tXpX = predicate_k(tXcX, limit=H)

        # Phase 1: Load input from global to register
        cute.copy(copy_atom_load, tXgX, tXrX, pred=tXpX)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum(x_sq, threads_per_row, reduction_buffer)

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # Phase 2: Load weight from global to register
        cute.copy(copy_atom_load, tXgW, tXrW, pred=tXpX)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias)) * inv_scale

        # Phase 3: Clamp and store to FP8 output using PTX scalar stores
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
def _get_compiled_rmsnorm_kernel(
    dtype_str: str, H: int, weight_bias: float, enable_pdl: bool
):
    """Get a compiled RMSNorm kernel using TVM-FFI."""
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = RMSNormKernel(dtype, H, weight_bias)

    sym_m = cute.sym_int()

    # Row stride alignment depends on H: for compact (M, H) with stride (H, 1),
    # each row starts at H * elem_bytes from the previous. The effective alignment
    # is gcd(base_align, H * elem_bytes). We use this so the compiler can verify
    # the copy atom's alignment requirement.
    elem_bytes = dtype.width // 8
    row_bytes = H * elem_bytes
    tensor_align = math.gcd(128, row_bytes)

    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (H,), assumed_align=tensor_align
    )
    y_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int32(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


@functools.cache
def _get_compiled_qk_rmsnorm_kernel(
    dtype_str: str, head_dim: int, weight_bias: float, num_warps: int, enable_pdl: bool
):
    """Get a compiled QKRMSNorm kernel for 3D tensors with arbitrary stride."""
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = QKRMSNormKernel(dtype, head_dim, weight_bias, num_warps)

    # Use symbolic sizes for B, N dimensions
    sym_b = cute.sym_int()
    sym_n = cute.sym_int()

    # Use symbolic strides for arbitrary stride support
    # stride[-1] must be 1 (contiguous in head_dim), but batch/head strides can be anything
    sym_batch_stride_x = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_head_stride_x = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_batch_stride_y = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_head_stride_y = cute.sym_int(divisibility=kernel_obj.vec_size)

    # Create 3D fake tensors with arbitrary stride
    x_fake = cute.runtime.make_fake_tensor(
        dtype,
        (sym_b, sym_n, head_dim),
        (sym_batch_stride_x, sym_head_stride_x, 1),
        assumed_align=16,
    )
    y_fake = cute.runtime.make_fake_tensor(
        dtype,
        (sym_b, sym_n, head_dim),
        (sym_batch_stride_y, sym_head_stride_y, 1),
        assumed_align=16,
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (head_dim,), assumed_align=16)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Compile with TVM-FFI enabled
    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int32(1),  # Dummy B
        Int32(1),  # Dummy N
        Float32(1e-6),  # Dummy eps
        enable_pdl,
        Int32(1),  # Dummy num_blocks
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


@functools.cache
def _get_compiled_rmsnorm_quant_kernel(
    dtype_str: str,
    out_dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    use_hw_fp8: bool = True,
):
    """Get a compiled RMSNorm + Quant kernel using TVM-FFI."""
    dtype = get_cutlass_dtype(dtype_str)
    out_dtype = get_cutlass_dtype(out_dtype_str)
    kernel_obj = RMSNormQuantKernel(dtype, H, weight_bias, use_hw_fp8=use_hw_fp8)

    sym_m = cute.sym_int()
    sym_row_stride_x = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_row_stride_y = cute.sym_int(divisibility=kernel_obj.vec_size_out)

    x_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    y_fake = cute.runtime.make_fake_tensor(
        out_dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
    )
    s_fake = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=4)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int32(1),
        s_fake,
        Float32(1e-6),  # eps
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


# =============================================================================
# CuTe DSL API Functions
# =============================================================================


def rmsnorm_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL RMSNorm implementation.

    Input and output must be contiguous (row-major). Uses async global→shared
    copy with multi-row blocks for improved wave utilization.
    """
    shape = input.shape
    H = shape[-1]

    if len(shape) == 3:
        M = shape[0] * shape[1]
        input_2d = input.reshape(M, H).contiguous()
        out_2d = out.reshape(M, H)
    else:
        M = shape[0]
        input_2d = input.contiguous()
        out_2d = out

    kernel = _get_compiled_rmsnorm_kernel(
        _torch_dtype_to_str(input.dtype), H, weight_bias, enable_pdl
    )
    kernel(input_2d, weight.contiguous(), out_2d, M, eps)


def qk_rmsnorm_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL QKRMSNorm for 3D tensors [batch, heads, head_dim].

    Supports arbitrary stride - no need to call contiguous().
    Each warp processes one (batch, head) pair independently using warp-only reduction.

    Args:
        input: Input tensor of shape [batch_size, num_heads, head_dim].
            Last dimension must be contiguous (stride[-1] == 1).
        weight: Weight tensor of shape [head_dim].
        output: Output tensor (same shape as input).
        eps: Small constant for numerical stability.
        weight_bias: Bias added to weight (0 for standard RMSNorm, 1 for Gemma).
        enable_pdl: Enable Programmatic Dependent Launch for SM90+ GPUs.
    """
    shape = input.shape
    assert len(shape) == 3, "QKRMSNorm expects 3D input [batch, heads, head_dim]"

    batch_size, num_heads, head_dim = shape
    M = batch_size * num_heads

    # Kernel configuration
    num_warps = 4

    # Calculate grid size based on SM count and estimated occupancy
    num_sms = get_num_sm(input.device)
    blocks_per_sm = 16  # Theoretical max for 128-thread blocks
    max_blocks = num_sms * blocks_per_sm
    needed_blocks = (M + num_warps - 1) // num_warps
    num_blocks = min(max_blocks, needed_blocks)

    dtype_str = _torch_dtype_to_str(input.dtype)
    kernel = _get_compiled_qk_rmsnorm_kernel(
        dtype_str, head_dim, weight_bias, num_warps, enable_pdl
    )

    # Pass 3D tensors directly - kernel handles arbitrary stride
    kernel(input, weight, output, batch_size, num_heads, eps, num_blocks)


def rmsnorm_quant_cute(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL RMSNorm + FP8 quantization implementation.

    Supports arbitrary stride - no need to call contiguous().
    Last dimension must be contiguous (stride[-1] == 1).
    """

    shape = input.shape
    H = shape[-1]
    M = shape[0]

    dtype_str = _torch_dtype_to_str(input.dtype)
    out_dtype_str = _torch_dtype_to_str(out.dtype)
    kernel = _get_compiled_rmsnorm_quant_kernel(
        dtype_str,
        out_dtype_str,
        H,
        weight_bias,
        enable_pdl,
        use_hw_fp8=has_hw_fp8_cvt(input.device),
    )
    kernel(input, weight, out, M, scale, eps)


__all__ = [
    # Kernel classes
    "RMSNormKernel",
    "QKRMSNormKernel",
    "RMSNormQuantKernel",
    # Compiled kernel getters
    "_get_compiled_rmsnorm_kernel",
    "_get_compiled_qk_rmsnorm_kernel",
    "_get_compiled_rmsnorm_quant_kernel",
    # CuTe DSL APIs
    "rmsnorm_cute",
    "qk_rmsnorm_cute",
    "rmsnorm_quant_cute",
]
