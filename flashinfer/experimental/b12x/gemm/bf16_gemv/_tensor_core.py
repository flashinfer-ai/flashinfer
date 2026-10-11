"""Tensor-core GEMV for up to eight BF16 rows.

At decode row counts the projection streams the weight once, so its cost is
the weight read.  Each CTA owns 16 output features and its warps split K into
64-wide chunks.  Weights are the ``m16n8k16`` MMA's 16-row A operand and the
live rows its 8-column B operand, so one weight fragment serves every row.
The warp count evenly divides K/64, with at most eight warps per CTA.
Products accumulate in FP32; the warps reduce through shared memory in a fixed
order and the result is rounded once, so the output is deterministic.

Within a chunk, K is permuted identically for both operands so that each
thread's fragments come from 32 contiguous bytes (two 16-byte loads); the
chunk sum covers the same products.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    get_ptr_as_int64,
    ld_global_nc_v4_u32,
)

TC_MAX_ROWS = 8
TC_MAX_WARPS = 8
TC_K_MULTIPLE = 64
_LANES = 32


@cute.jit
def _flat(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout((Int64(1) << Int64(50),)))


class TensorCoreGemvKernel:
    """``out[m, n] = sum_k x[m, k] * w[n, k]`` for at most eight live rows."""

    def __init__(self, n: int, k: int):
        if k <= 0 or k % TC_K_MULTIPLE:
            raise ValueError(
                f"tensor-core GEMV requires K to be a positive multiple of {TC_K_MULTIPLE}"
            )
        self.n, self.k = int(n), int(k)
        chunks = self.k // TC_K_MULTIPLE
        self.warps = min(TC_MAX_WARPS, chunks)
        while chunks % self.warps:
            self.warps -= 1
        self.chunks_per_warp = chunks // self.warps

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "partials": cute.struct.Align[
                cute.struct.MemRange[Float32, self.warps * _LANES * 4], 16
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        bias: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
        x_column_stride: Int64,
        weight_column_stride: Int64,
        vector_loads: Int32,
        stream: cuda.CUstream,
    ):
        # Unit column strides and 16-byte rows are part of the eligibility
        # contract; bias is not (the selector keeps biased calls elsewhere).
        del bias, x_column_stride, weight_column_stride, vector_loads
        self.kernel(
            _flat(x),
            _flat(weight),
            _flat(output),
            rows,
            x_stride,
            weight_stride,
            output_stride,
        ).launch(
            grid=((self.n + 15) // 16, 1, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = Int32(thread) // _LANES
        lane = Int32(thread) % _LANES
        group = lane // 4
        quad = lane % 4
        first = Int32(block) * 16
        # Clamped loads keep partial tiles and absent rows in bounds; their
        # stores are masked.
        row_a = cutlass.min(first + group, Int32(self.n - 1))
        row_b = cutlass.min(first + group + 8, Int32(self.n - 1))
        token = cutlass.min(group, rows - 1)
        weight_base = get_ptr_as_int64(weight, Int64(0))
        x_base = get_ptr_as_int64(x, Int64(0))
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        for step in cutlass.range_constexpr(self.chunks_per_warp):
            chunk = warp * self.chunks_per_warp + step
            offset = Int64(chunk * 128 + quad * 32)
            pa = weight_base + Int64(row_a) * weight_stride * 2 + offset
            pb = weight_base + Int64(row_b) * weight_stride * 2 + offset
            px = x_base + Int64(token) * x_stride * 2 + offset
            wa0, wa1, wa2, wa3 = ld_global_nc_v4_u32(pa)
            wa4, wa5, wa6, wa7 = ld_global_nc_v4_u32(pa + Int64(16))
            wb0, wb1, wb2, wb3 = ld_global_nc_v4_u32(pb)
            wb4, wb5, wb6, wb7 = ld_global_nc_v4_u32(pb + Int64(16))
            x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
            x4, x5, x6, x7 = ld_global_nc_v4_u32(px + Int64(16))
            wa = (wa0, wa1, wa2, wa3, wa4, wa5, wa6, wa7)
            wb = (wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7)
            xs = (x0, x1, x2, x3, x4, x5, x6, x7)
            for s in cutlass.range_constexpr(4):
                acc0, acc1, acc2, acc3 = bf16_mma_m16n8k16_f32(
                    acc0,
                    acc1,
                    acc2,
                    acc3,
                    wa[2 * s],
                    wb[2 * s],
                    wa[2 * s + 1],
                    wb[2 * s + 1],
                    xs[2 * s],
                    xs[2 * s + 1],
                )

        storage = cutlass.utils.SmemAllocator().allocate(self._shared_storage_cls())
        partials = storage.partials.get_tensor(
            cute.make_layout((self.warps * _LANES * 4,))
        )
        base = (warp * _LANES + lane) * 4
        partials[base] = acc0
        partials[base + 1] = acc1
        partials[base + 2] = acc2
        partials[base + 3] = acc3
        cute.arch.sync_threads()
        if warp == 0:
            for e in cutlass.range_constexpr(4):
                total = Float32(0.0)
                for w in cutlass.range_constexpr(self.warps):
                    total = total + partials[Int32(w * _LANES * 4) + lane * 4 + e]
                feature = first + group + (8 if e >= 2 else 0)
                row = quad * 2 + (e % 2)
                if feature < self.n and row < rows:
                    output[Int64(row) * output_stride + Int64(feature)] = total.to(
                        BFloat16
                    )
