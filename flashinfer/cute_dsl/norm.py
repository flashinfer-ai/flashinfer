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

CuTe-DSL Norm Kernels
====================

High-performance normalization kernels implemented using NVIDIA CuTe-DSL.

Includes:
- RMSNormKernel: Basic RMSNorm (also handles Gemma variant with weight_bias=1.0)
- RMSNormQuantKernel: RMSNorm + FP8 quantization
- FusedAddRMSNormKernel: Fused residual add + RMSNorm
- FusedAddRMSNormQuantKernel: Fused residual add + RMSNorm + FP8 quantization
- LayerNormKernel: Traditional LayerNorm with mean and variance
"""

import functools
import math
import operator
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .utils import get_cutlass_dtype, get_num_sm


# =============================================================================
# Constants
# =============================================================================

FLOAT8_E4M3_MAX = 448.0  # Maximum value representable in FP8 E4M3
COPY_BITS = 128  # 128-bit vectorized loads


# =============================================================================
# PTX Intrinsics
# =============================================================================


@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """Fast reciprocal using PTX rcp.approx.ftz.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmin_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute min of two float32 values using PTX min.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max of two float32 values using PTX max.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_and_store_f32_to_e4m3(val: Float32, addr: Int64, *, loc=None, ip=None):
    """Convert float32 to E4M3 and store single byte to global memory.

    This handles the case where we need to store a single FP8 value,
    which can't be done with vectorized CuTe copies (min 16 bits).
    """
    llvm.inline_asm(
        None,  # void return type
        [Float32(val).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 fp8_pair;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $0;
            st.global.b8 [$1], fp8_pair;
        }
        """,
        "f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64."""
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


# =============================================================================
# Warp and Block Reduction Utilities
# =============================================================================


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """Reduce across threads in a warp using butterfly shuffle."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val


@cute.jit
def block_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """Block reduction across multiple warps using shared memory."""
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    num_warps = cute.size(reduction_buffer.shape)

    if lane_idx == 0:
        reduction_buffer[warp_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < num_warps:
        block_reduce_val = reduction_buffer[lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce_sum(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
) -> Float32:
    """Row reduction for sum operation."""
    local_val = x.reduce(
        cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
    )

    warp_width = min(threads_per_row, 32)
    warp_val = warp_reduce(local_val, operator.add, width=warp_width)

    warps_per_row = max(threads_per_row // 32, 1)

    if cutlass.const_expr(warps_per_row > 1):
        return block_reduce(warp_val, operator.add, reduction_buffer, Float32(0.0))
    else:
        return warp_val


# =============================================================================
# Predicate Utility
# =============================================================================


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking (2D tensors)."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][1], limit
            )
    return tXpX


@cute.jit
def predicate_k_3d(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking (3D tensors).

    For 3D tensors after local_tile, the last coordinate [2] is the head_dim dimension.
    """
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            # For 3D tensor, coordinate[2] is the head_dim index
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][2], limit
            )
    return tXpX


# =============================================================================
# Helper Functions for Kernel Configuration
# =============================================================================


def compute_optimal_vec_size(H: int, max_vec_size: int) -> int:
    """Compute vec_size that maximizes warp utilization.

    For small hidden sizes, using max vec_size may result in fewer than 32 threads,
    wasting warp resources. This function finds the largest vec_size that:
    1. Divides H evenly
    2. Results in at least 32 threads (one full warp)

    Examples:
    - H=128, max=8: vec_size=8 gives 16 threads, vec_size=4 gives 32 threads -> return 4
    - H=4096, max=8: vec_size=8 gives 512 threads -> return 8
    - H=111, max=8: no vec_size divides evenly with >=32 threads, use gcd -> return 1
    """
    # Try vec_sizes from largest to smallest
    for vec_size in [
        max_vec_size,
        max_vec_size // 2,
        max_vec_size // 4,
        max_vec_size // 8,
    ]:
        if vec_size < 1:
            continue
        if H % vec_size != 0:
            continue
        threads_needed = H // vec_size
        if threads_needed >= 32:
            return vec_size
    # Fallback: use gcd for correctness (handles odd sizes like 111)
    return math.gcd(max_vec_size, H)


def compute_threads_per_row(H: int, vec_size: int) -> int:
    """Compute optimal threads per row based on hidden size."""
    threads_needed = (H + vec_size - 1) // vec_size
    # Round up to power of 2, capped at 1024
    threads = 32
    while threads < threads_needed and threads < 1024:
        threads *= 2
    return min(threads, 1024)


def make_tv_layout(threads_per_row: int, vec_size: int, num_vec_blocks: int) -> tuple:
    """Create Thread-Value layout for coalesced vectorized memory access.

    This layout distributes work across threads where each thread handles
    vec_size consecutive elements, and threads are arranged for coalesced access.

    Args:
        threads_per_row: Number of threads processing one row
        vec_size: Number of elements each thread processes per vector load
        num_vec_blocks: Number of vector blocks per row

    Returns:
        Tuple of (shape, stride) for creating cute.Layout
    """
    shape = (
        (threads_per_row, 1),
        (vec_size, num_vec_blocks),
    )
    stride = (
        (vec_size, 1),
        (1, vec_size * threads_per_row),
    )
    return shape, stride


# =============================================================================
# RMSNormKernel
# =============================================================================


class RMSNormKernel:
    """
    RMSNorm Kernel using CuTe-DSL.

    Computes: output = input / sqrt(mean(input^2) + eps) * (weight + weight_bias)

    Key optimizations:
    1. 128-bit vectorized loads for input and weight
    2. Two-stage reduction: warp shuffle + cross-warp shared memory
    3. All computations in FP32 for numerical stability
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
        max_vec_size = COPY_BITS // elem_bits  # 8 for float16/bfloat16, 4 for float32
        self.vec_size = compute_optimal_vec_size(H, max_vec_size)
        self.copy_bits = self.vec_size * elem_bits  # Actual bits per copy

        # Thread configuration
        self.threads_per_row = compute_threads_per_row(H, self.vec_size)
        self.num_threads = self.threads_per_row  # One row per block
        self.num_warps = max(self.threads_per_row // 32, 1)

        # Vectorization blocks
        self.num_vec_blocks = max(
            1, (H // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

    def _smem_size_in_bytes(self) -> int:
        """Calculate shared memory requirement."""
        # Only reduction buffer needed (no shared memory for input/weight)
        return self.num_warps * 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        eps: Float32,
        stream,
    ):
        """Launch the RMSNorm kernel."""
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (1, self.cols_per_tile)

        self.kernel(mX, mW, mY, M, eps, tv_layout, tiler_mn).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        """Device kernel for RMSNorm."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        H = self.H
        weight_bias = self.weight_bias
        threads_per_row = tv_layout.shape[0][0]
        num_warps = self.num_warps
        copy_bits = self.copy_bits

        # Allocate shared memory (only reduction buffer needed)
        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((num_warps,)),
            byte_alignment=4,
        )

        # Create identity tensor for coordinate tracking
        idX = cute.make_identity_tensor(mX.shape)

        # Slice for this row
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # Expand weight to 2D for consistent tiling
        mW_2d = cute.prepend_ones(mW, up_to_rank=2)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        # Create TiledCopy for load and store (both use CopyUniversalOp for sync operations)
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn)
        thr_copy = tiled_copy.get_slice(tidx)

        # Partition tensors
        tXgX = thr_copy.partition_S(gX)
        tXgW = thr_copy.partition_S(gW)
        tXgY = thr_copy.partition_D(gY)
        tXcX = thr_copy.partition_S(cX)

        # Register fragments - initialize to zero for proper handling of out-of-bounds threads
        tXrX = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
        tXrW = cute.make_rmem_tensor(tXgW.shape, mW.element_type)
        tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
        tXrW.store(cute.zeros_like(tXrW, dtype=mW.element_type))

        # Bounds checking (column boundary only, row is always valid since grid=[M,1,1])
        tXpX = predicate_k(tXcX, limit=H)

        # ===================================================================
        # Phase 1: Load input from global to register
        # ===================================================================
        cute.copy(copy_atom, tXgX, tXrX, pred=tXpX)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum(x_sq, threads_per_row, reduction_buffer)

        # Compute rstd = 1 / sqrt(mean(x^2) + eps)
        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # ===================================================================
        # Phase 2: Load weight from global to register
        # ===================================================================
        cute.copy(copy_atom, tXgW, tXrW, pred=tXpX)

        w = tXrW.load().to(Float32)

        # output = input * rstd * (weight + weight_bias)
        y = x * rstd * (w + Float32(weight_bias))

        # Store output using cute.copy with predicate
        tYrY = y.to(mY.element_type)
        tXrY = cute.make_rmem_tensor(tXgY.shape, mY.element_type)
        tXrY.store(tYrY)

        cute.copy(copy_atom, tXrY, tXgY, pred=tXpX)


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

        # Create 2D identity tensor for bounds checking
        id2d = cute.make_identity_tensor((1, head_dim))

        # Each warp processes multiple rows with grid-stride loop
        row_idx = worker_idx
        while row_idx < M:
            batch_idx = row_idx // N
            head_idx = row_idx % N

            # Use slice to get 1D row, then prepend_ones to make 2D
            # local_tile uses mX's stride to compute correct address
            gX_row = cute.local_tile(
                mX, (1, 1, self.cols_per_tile), (batch_idx, head_idx, 0)
            )
            gY_row = cute.local_tile(
                mY, (1, 1, self.cols_per_tile), (batch_idx, head_idx, 0)
            )

            # Flatten the first two dims (both size 1) to get 2D tensor for tiled_copy
            gX = cute.make_tensor(
                gX_row.iterator,
                cute.make_layout(
                    (1, self.cols_per_tile), stride=(self.cols_per_tile, 1)
                ),
            )
            gY = cute.make_tensor(
                gY_row.iterator,
                cute.make_layout(
                    (1, self.cols_per_tile), stride=(self.cols_per_tile, 1)
                ),
            )
            cX = cute.local_tile(id2d, tiler_2d, (0, 0))
            gW = cute.local_tile(mW_2d, tiler_2d, (0, 0))

            # Partition tensors for this thread
            tXgX = thr_copy.partition_S(gX)
            tXgW = thr_copy.partition_S(gW)
            tXgY = thr_copy.partition_D(gY)
            tXcX = thr_copy.partition_S(cX)

            # Register fragments - initialize to zero
            tXrX = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
            tXrW = cute.make_rmem_tensor(tXgW.shape, mW.element_type)
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tXrW.store(cute.zeros_like(tXrW, dtype=mW.element_type))

            # Bounds checking predicate (2D)
            tXpX = predicate_k(tXcX, limit=head_dim)

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

            # Phase 2: Load weight, normalize, and store
            cute.copy(copy_atom, tXgW, tXrW, pred=tXpX)

            w = tXrW.load().to(Float32)

            # output = input * rstd * (weight + weight_bias)
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
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias

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
        scale: Float32,
        eps: Float32,
        stream,
    ):
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (1, self.cols_per_tile)

        self.kernel(mX, mW, mY, M, scale, eps, tv_layout, tiler_mn).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        scale: Float32,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

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
            Float32, cute.make_layout((num_warps,)), byte_alignment=4
        )

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        mW_2d = cute.prepend_ones(mW, up_to_rank=2)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits
        )

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        tXgX = thr_copy_load.partition_S(gX)
        tXgW = thr_copy_load.partition_S(gW)
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
                    clamped = fmax_f32(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                    clamped = fmin_f32(clamped, Float32(FLOAT8_E4M3_MAX))
                    # Use PTX to convert and store FP8 byte
                    out_offset = bidx * H + idx
                    out_ptr = get_ptr_as_int64(mY, Int32(out_offset))
                    cvt_and_store_f32_to_e4m3(clamped, out_ptr)


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
        stream,
    ):
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (1, self.cols_per_tile)

        self.kernel(mX, mR, mW, M, eps, tv_layout, tiler_mn).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int32,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

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
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn)
        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXgR = thr_copy.partition_S(gR)
        tXgW = thr_copy.partition_S(gW)
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
        stream,
    ):
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (1, self.cols_per_tile)

        self.kernel(mY, mX, mR, mW, M, scale, eps, tv_layout, tiler_mn).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
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
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

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
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        tXgX = thr_copy_load.partition_S(gX)
        tXgR = thr_copy_load.partition_S(gR)
        tXgW = thr_copy_load.partition_S(gW)
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
                    clamped = fmax_f32(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                    clamped = fmin_f32(clamped, Float32(FLOAT8_E4M3_MAX))
                    # Use PTX to convert and store FP8 byte
                    out_offset = bidx * H + idx
                    out_ptr = get_ptr_as_int64(mY, Int32(out_offset))
                    cvt_and_store_f32_to_e4m3(clamped, out_ptr)


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
            tv_layout,
            tiler_mn,
            tv_layout_f32,
            tiler_mn_f32,
        ).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
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
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        tv_layout_f32: cute.Layout,
        tiler_mn_f32: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

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
        gGamma = cute.local_tile(mGamma_2d, tiler_mn_f32, (0, 0))

        mBeta_2d = cute.prepend_ones(mBeta, up_to_rank=2)
        gBeta = cute.local_tile(mBeta_2d, tiler_mn_f32, (0, 0))

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
        tGgGamma = thr_copy_load_f32.partition_S(gGamma)
        tGsGamma = thr_copy_load_f32.partition_D(sGamma_f32)
        tGgBeta = thr_copy_load_f32.partition_S(gBeta)
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


# =============================================================================
# Python API Functions - Using TVM-FFI Compilation Pattern
# =============================================================================


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float8_e4m3fn: "float8_e4m3fn",
    }
    return dtype_map[dtype]


@functools.cache
def _get_compiled_rmsnorm_kernel(dtype_str: str, H: int, weight_bias: float):
    """Get a compiled RMSNorm kernel using TVM-FFI."""
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = RMSNormKernel(dtype, H, weight_bias)

    # Use symbolic size for dynamic M dimension
    sym_m = cute.sym_int()
    # Use symbolic stride for arbitrary row stride (last dim must be contiguous)
    sym_row_stride_x = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_row_stride_y = cute.sym_int(divisibility=kernel_obj.vec_size)

    # Create fake tensors with symbolic stride for arbitrary stride support
    x_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    y_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
    )

    # Create fake stream that uses environment stream at runtime
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Compile with TVM-FFI enabled
    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int32(1),  # Dummy M
        Float32(1e-6),  # Dummy eps
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        input: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        M: int,
        eps: float,
    ) -> None:
        compiled_kernel(
            input,
            weight,
            out,
            Int32(M),
            Float32(eps),
        )

    return tensor_api


@functools.cache
def _get_compiled_qk_rmsnorm_kernel(
    dtype_str: str, head_dim: int, weight_bias: float, num_warps: int
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
        False,  # enable_pdl
        Int32(1),  # Dummy num_blocks
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        B: int,
        N: int,
        eps: float,
        num_blocks: int,
    ) -> None:
        compiled_kernel(
            input,
            weight,
            output,
            Int32(B),
            Int32(N),
            Float32(eps),
            Int32(num_blocks),
        )

    return tensor_api


@functools.cache
def _get_compiled_rmsnorm_quant_kernel(
    dtype_str: str, out_dtype_str: str, H: int, weight_bias: float
):
    """Get a compiled RMSNorm + Quant kernel using TVM-FFI."""
    dtype = get_cutlass_dtype(dtype_str)
    out_dtype = get_cutlass_dtype(out_dtype_str)
    kernel_obj = RMSNormQuantKernel(dtype, H, weight_bias)

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

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int32(1),
        Float32(1.0),  # scale
        Float32(1e-6),  # eps
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        M: int,
        scale: float,
        eps: float,
    ) -> None:
        compiled_kernel(
            input,
            weight,
            out,
            Int32(M),
            Float32(scale),
            Float32(eps),
        )

    return tensor_api


@functools.cache
def _get_compiled_fused_add_rmsnorm_kernel(dtype_str: str, H: int, weight_bias: float):
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
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        M: int,
        eps: float,
    ) -> None:
        compiled_kernel(
            input,
            residual,
            weight,
            Int32(M),
            Float32(eps),
        )

    return tensor_api


@functools.cache
def _get_compiled_fused_add_rmsnorm_quant_kernel(
    dtype_str: str, out_dtype_str: str, H: int, weight_bias: float
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
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        out: torch.Tensor,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        M: int,
        scale: float,
        eps: float,
    ) -> None:
        compiled_kernel(
            out,
            input,
            residual,
            weight,
            Int32(M),
            Float32(scale),
            Float32(eps),
        )

    return tensor_api


@functools.cache
def _get_compiled_layernorm_kernel(dtype_str: str, gamma_dtype_str: str, H: int):
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


def rmsnorm_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL RMSNorm implementation.

    Supports arbitrary stride - no need to call contiguous().
    Last dimension must be contiguous (stride[-1] == 1).
    """
    H = input.shape[-1]
    if input.dim() == 3:
        M = input.shape[0] * input.shape[1]
        input_2d = input.view(M, H)
        out_2d = out.view(M, H)
    else:
        M = input.shape[0]
        input_2d = input
        out_2d = out

    dtype_str = _torch_dtype_to_str(input.dtype)
    kernel = _get_compiled_rmsnorm_kernel(dtype_str, H, weight_bias)
    kernel(input_2d, weight, out_2d, M, eps)


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
    assert input.dim() == 3, "QKRMSNorm expects 3D input [batch, heads, head_dim]"

    batch_size, num_heads, head_dim = input.shape
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
        dtype_str, head_dim, weight_bias, num_warps
    )

    # Pass 3D tensors directly - kernel handles arbitrary stride
    kernel(input, weight, output, batch_size, num_heads, eps, num_blocks)


def rmsnorm_quant_cute(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL RMSNorm + FP8 quantization implementation.

    Supports arbitrary stride - no need to call contiguous().
    Last dimension must be contiguous (stride[-1] == 1).
    """

    H = input.shape[-1]
    M = input.shape[0]

    dtype_str = _torch_dtype_to_str(input.dtype)
    out_dtype_str = _torch_dtype_to_str(out.dtype)
    kernel = _get_compiled_rmsnorm_quant_kernel(
        dtype_str, out_dtype_str, H, weight_bias
    )
    kernel(out, input, weight, M, scale, eps)


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
    kernel = _get_compiled_fused_add_rmsnorm_kernel(dtype_str, H, weight_bias)
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
        dtype_str, out_dtype_str, H, weight_bias
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


def layernorm_cute(
    out: torch.Tensor,
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """CuTe DSL LayerNorm implementation.

    Supports arbitrary stride - no need to call contiguous().
    Last dimension must be contiguous (stride[-1] == 1).
    """

    H = input.shape[-1]
    M = input.shape[0]

    dtype_str = _torch_dtype_to_str(input.dtype)
    gamma_dtype_str = _torch_dtype_to_str(gamma.dtype)
    kernel = _get_compiled_layernorm_kernel(dtype_str, gamma_dtype_str, H)
    kernel(out, input, gamma, beta, M, eps)
