"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Small-N bf16 GEMV for decode-size m on narrow projections, where cuBLAS
# tile kernels are mostly launch and tile overhead.

import functools
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cuda.bindings.driver as cuda
from cutlass import Int64
from cutlass.cutlass_dsl import Int32, T, Uint32, dsl_user_op
from cutlass._mlir.dialects import llvm

_THREADS = 128

#: Largest m routed to this kernel. Per-thread work grows linearly in m, so
#: the advantage over cuBLAS fades beyond a few rows.
SMALL_M_MAX = 8


# Vendored from flashinfer.cute_dsl.fp4_common so this file is self-contained
# and its content alone keys the disk cache.


@dsl_user_op
def _get_ptr_as_int64(tensor: cute.Tensor, offset, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64."""
    elem_ptr = tensor.iterator + offset
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@dsl_user_op
def _ld_global_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)


@dsl_user_op
def _u32_as_f32(value: Uint32, *, loc=None, ip=None) -> cutlass.Float32:
    """Bitcast a uint32 to float32 (mov.b32, no numeric conversion)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _warp_reduce(val, op):
    """Reduce a scalar across the 32 threads of a warp (butterfly shuffle)."""
    for i in cutlass.range_constexpr(5):  # log2(32)
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def _block_reduce(val, op, reduction_buffer: cute.Tensor, init_val):
    """Block reduction across warps using shared memory."""
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    num_warps = cute.size(reduction_buffer.shape[1])

    if lane_idx == 0:
        reduction_buffer[0, warp_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < num_warps:
        block_reduce_val = reduction_buffer[0, lane_idx]
    return _warp_reduce(block_reduce_val, op)


def _fadd(a, b):
    return a + b


class SmallNGemvKernel:
    """One CTA per output column: ``y[0:m, n] = x[0:m, :] @ w[n, :]``.

    128 threads stride over K with 128-bit loads, each keeping ``m`` f32
    accumulators. Accumulation is f32 like cuBLAS but in a different
    reduction order, so results match to bf16 rounding, not bitwise.
    """

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        y: cute.Tensor,
        stream: cuda.CUstream,
    ):
        n = w.shape[0]
        self.kernel(x, w, y).launch(
            grid=(n, 1, 1),
            block=[_THREADS, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        m = mX.shape[0]  # static (compile key)
        n = mW.shape[0]  # static (compile key)
        k = Int32(mX.shape[1])
        nvec = k // Int32(8)  # 8 bf16 per 128-bit load
        # Offsets are computed in Int64 so an id * stride product cannot wrap
        # before _get_ptr_as_int64 widens it into the pointer.
        w_base = Int64(bidx) * Int64(k)

        smem = cutlass.utils.SmemAllocator()
        red_buf = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((1, _THREADS // 32)),
            byte_alignment=16,
        )

        acc = cute.make_rmem_tensor((m,), cutlass.Float32)
        for r in cutlass.range_constexpr(m):
            acc[r] = cutlass.Float32(0.0)

        # Each thread reads its 8-wide chunk of the W row once and dots it
        # against the same chunk of every x row. bf16 -> f32 is a shift (low
        # half) or mask (high half) bitcast, exact by construction.
        i = Int32(tidx)
        while i < nvec:
            col = Int64(i * Int32(8))
            w0, w1, w2, w3 = _ld_global_v4_u32(_get_ptr_as_int64(mW, w_base + col))
            for r in cutlass.range_constexpr(m):
                x0, x1, x2, x3 = _ld_global_v4_u32(
                    _get_ptr_as_int64(mX, Int64(r) * Int64(k) + col)
                )
                s = acc[r]
                for wv, xv in ((w0, x0), (w1, x1), (w2, x2), (w3, x3)):
                    s = s + _u32_as_f32(wv << Uint32(16)) * _u32_as_f32(
                        xv << Uint32(16)
                    )
                    s = s + _u32_as_f32(wv & Uint32(0xFFFF0000)) * _u32_as_f32(
                        xv & Uint32(0xFFFF0000)
                    )
                acc[r] = s
            i += Int32(_THREADS)

        # One reduction per output row. red_buf is reused, so a barrier keeps
        # iteration r+1's lane-0 writes from racing iteration r's reads.
        for r in cutlass.range_constexpr(m):
            v = _warp_reduce(acc[r], _fadd)
            total = _block_reduce(v, _fadd, red_buf, cutlass.Float32(0.0))
            if tidx == Int32(0):
                # mY is the flat (m*n,) view: a 2D (m, n) tensor degenerates
                # to all-stride-1 at n == 1 and breaks dlpack layout deduction.
                mY[Int32(r * n) + bidx] = cutlass.BFloat16(total)
            cute.arch.barrier()


@functools.cache
def get_bf16_gemv_kernel(m: int, n: int, k: int, device_index: int = 0):
    """Compiled small-N bf16 GEMV for ``(m, n, k)``, disk-cached.

    Returns a TVM-FFI callable ``kernel(x, w, y_flat)`` where ``x`` is
    ``(m, k)`` bf16 contiguous, ``w`` is ``(n, k)`` bf16 contiguous
    (row-major weight, as ``F.linear`` takes it), and ``y_flat`` the
    flattened ``(m * n,)`` view of a preallocated bf16 output. Launches on
    the TVM-FFI environment stream (the current torch stream).

    ``device_index`` keys the in-memory cache per device. The disk cache is
    keyed by architecture.
    """
    assert 1 <= m <= SMALL_M_MAX, f"small-N GEMV requires m<={SMALL_M_MAX}, got {m}"
    assert k % 8 == 0, f"K must be a multiple of 8, got {k}"
    assert n >= 1

    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    def compile_kernel():
        x_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16
        )
        w_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (n, k), stride_order=(1, 0), assumed_align=16
        )
        y_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (m * n,), assumed_align=16
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            SmallNGemvKernel(),
            x_fake,
            w_fake,
            y_fake,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return build_and_load_cute_dsl_kernel(
        "bf16_gemv",
        f"m{m}_n{n}_k{k}",
        compile_kernel,
        extra_key_files=(__file__,),
    )
