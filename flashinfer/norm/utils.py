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

Shared CuTe DSL Utilities for Norm Kernels
==========================================

Common utilities used by all norm kernels:
- Constants for vectorization and FP8 quantization
- PTX intrinsics for fast reciprocal and FP8 conversion
- Warp and block reduction utilities
- Predicate helpers for bounds checking
- Layout configuration helpers
- Type conversion utilities
"""

import math
import operator
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from ..cute_dsl.utils import get_cutlass_dtype, get_num_sm


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
# Type Conversion Utilities
# =============================================================================


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float8_e4m3fn: "float8_e4m3fn",
    }
    return dtype_map[dtype]


# Re-export utilities from cute_dsl.utils for convenience
__all__ = [
    # Constants
    "FLOAT8_E4M3_MAX",
    "COPY_BITS",
    # PTX intrinsics
    "rcp_approx_ftz",
    "cvt_and_store_f32_to_e4m3",
    "get_ptr_as_int64",
    # Reduction utilities
    "warp_reduce",
    "block_reduce",
    "row_reduce_sum",
    # Predicate utilities
    "predicate_k",
    "predicate_k_3d",
    # Configuration helpers
    "compute_optimal_vec_size",
    "compute_threads_per_row",
    "make_tv_layout",
    # Type utilities
    "_torch_dtype_to_str",
    # Re-exports from cute_dsl.utils
    "get_cutlass_dtype",
    "get_num_sm",
]
