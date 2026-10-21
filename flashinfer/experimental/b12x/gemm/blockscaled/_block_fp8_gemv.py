"""Small-row GEMV for serialized 128x128 block-FP8 weights.

At decode row counts (at most eight rows) the dense block-FP8 GEMM is
latency-bound: it stages one weight tile per CTA and reduces split-K partials
in a second pass.  Here each CTA owns 16 output features and its 8 warps split
K by whole 128-blocks.  Weights are the MMA's 16-row A operand and the rows its
8-column B operand, through the same unit-scale MXF8 MMA the dense kernel uses.
Each 128-block sum is scaled in FP32 by ``rhs_scale[n // 128, kb] *
lhs_scale[m, kb]`` and accumulated in FP32; the warps reduce through shared
memory in a fixed order and the result is rounded to BF16 once, so the output
is deterministic.

Within a 128-block, K is permuted identically for both operands so that each
thread's fragments come from 32 contiguous bytes (two 16-byte loads); every
block sum still covers the same products.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.intrinsics.memory import get_ptr_as_int64, ld_global_nc_v4_u32
from b12x._lib.intrinsics.mma import mxfp8_mma_m16n8k32_f32_e4m3
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import cuda_stream_from_int_or_current, current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.gemm.bf16_gemv._kernel import _flat

MAX_ROWS = 8
# Measured on SM120 (benchmarks/experimental/b12x/benchmark_block_fp8_gemv.py, cold L2): 1.1-3x
# faster than the dense GEMM up to 4096 output features for K in 1536..7168;
# from 8192 both stream weights at the same rate and the dense GEMM keeps them.
MAX_OUT_FEATURES = 4096
_WARPS = 8
_LANES = 32
_UNIT_E8M0 = 0x7F7F7F7F  # four E8M0 exponents of 2^0
_TYPES = (Uint8, Float32, Uint8, Float32, BFloat16)
_ALIGN = (16, 4, 16, 4, 2)


def supports(rows: int, out_features: int, in_features: int) -> bool:
    """True when the GEMV serves ``rows`` x ``in_features`` -> ``out_features``."""
    return (
        1 <= rows <= MAX_ROWS
        and in_features > 0
        and in_features % 128 == 0
        and 0 < out_features <= MAX_OUT_FEATURES
    )


class BlockFP8Gemv:
    """``out[m, n] = sum_kb (lhs[m, kb] . rhs[n, kb]) * lhs_scale[m, kb] * rhs_scale[n // 128, kb]``."""

    def __init__(self, n: int, k: int):
        if k % 128:
            raise ValueError("block-FP8 GEMV requires K to be a multiple of 128")
        self.n, self.k = int(n), int(k)
        self.blocks = self.k // 128
        self.blocks_per_warp = (self.blocks + _WARPS - 1) // _WARPS

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "partials": cute.struct.Align[
                cute.struct.MemRange[Float32, _WARPS * _LANES * 4], 16
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        lhs: cute.Pointer,
        lhs_scale: cute.Pointer,
        rhs: cute.Pointer,
        rhs_scale: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(lhs),
            _flat(lhs_scale),
            _flat(rhs),
            _flat(rhs_scale),
            _flat(output),
            rows,
        ).launch(
            grid=((self.n + 15) // 16, 1, 1),
            block=(_WARPS * _LANES, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        lhs: cute.Tensor,
        lhs_scale: cute.Tensor,
        rhs: cute.Tensor,
        rhs_scale: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = Int32(thread) // _LANES
        lane = Int32(thread) % _LANES
        group = lane // 4
        quad = lane % 4
        first = Int32(block) * 16
        # Clamped loads keep partial tiles in bounds; their stores are masked.
        row_a = cutlass.min(first + group, Int32(self.n - 1))
        row_b = cutlass.min(first + group + 8, Int32(self.n - 1))
        token = cutlass.min(group, rows - 1)
        col0 = cutlass.min(quad * 2, rows - 1)
        col1 = cutlass.min(quad * 2 + 1, rows - 1)
        rhs_base = get_ptr_as_int64(rhs, Int64(0))
        lhs_base = get_ptr_as_int64(lhs, Int64(0))
        k = Int64(self.k)
        # 16 features never straddle a 128-row scale block.
        scale_row = Int64(first // 128) * self.blocks
        unit = Uint32(_UNIT_E8M0)
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        for step in cutlass.range_constexpr(self.blocks_per_warp):
            kb = warp * self.blocks_per_warp + step
            if kb < self.blocks:
                offset = Int64(kb * 128 + quad * 32)
                pa = rhs_base + Int64(row_a) * k + offset
                pb = rhs_base + Int64(row_b) * k + offset
                px = lhs_base + Int64(token) * k + offset
                wa0, wa1, wa2, wa3 = ld_global_nc_v4_u32(pa)
                wa4, wa5, wa6, wa7 = ld_global_nc_v4_u32(pa + Int64(16))
                wb0, wb1, wb2, wb3 = ld_global_nc_v4_u32(pb)
                wb4, wb5, wb6, wb7 = ld_global_nc_v4_u32(pb + Int64(16))
                x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                x4, x5, x6, x7 = ld_global_nc_v4_u32(px + Int64(16))
                wa = (wa0, wa1, wa2, wa3, wa4, wa5, wa6, wa7)
                wb = (wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7)
                xs = (x0, x1, x2, x3, x4, x5, x6, x7)
                d0 = Float32(0.0)
                d1 = Float32(0.0)
                d2 = Float32(0.0)
                d3 = Float32(0.0)
                for s in cutlass.range_constexpr(4):
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        d0,
                        d1,
                        d2,
                        d3,
                        wa[2 * s],
                        wb[2 * s],
                        wa[2 * s + 1],
                        wb[2 * s + 1],
                        xs[2 * s],
                        xs[2 * s + 1],
                        unit,
                        unit,
                    )
                weight_scale = rhs_scale[scale_row + Int64(kb)]
                s0 = weight_scale * lhs_scale[Int64(col0) * self.blocks + Int64(kb)]
                s1 = weight_scale * lhs_scale[Int64(col1) * self.blocks + Int64(kb)]
                acc0 = acc0 + d0 * s0
                acc1 = acc1 + d1 * s1
                acc2 = acc2 + d2 * s0
                acc3 = acc3 + d3 * s1

        storage = cutlass.utils.SmemAllocator().allocate(self._shared_storage_cls())
        partials = storage.partials.get_tensor(cute.make_layout((_WARPS * _LANES * 4,)))
        base = (warp * _LANES + lane) * 4
        partials[base] = acc0
        partials[base + 1] = acc1
        partials[base + 2] = acc2
        partials[base + 3] = acc3
        cute.arch.sync_threads()
        if warp == 0:
            for e in cutlass.range_constexpr(4):
                total = Float32(0.0)
                for w in cutlass.range_constexpr(_WARPS):
                    total = total + partials[Int32(w * _LANES * 4) + lane * 4 + e]
                feature = first + group + (8 if e >= 2 else 0)
                row = quad * 2 + (e % 2)
                if feature < self.n and row < rows:
                    output[Int64(row) * self.n + Int64(feature)] = total.to(BFloat16)


@program_cache(scope="preparation")
def compile_block_fp8_gemv(ordinal: int, n: int, k: int):
    fake = tuple(
        make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)
        for dtype, align in zip(_TYPES, _ALIGN, strict=True)
    )
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            BlockFP8Gemv(n, k),
            *fake,
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.block_fp8.gemv", 2, (ordinal, n, k)
            ),
        )

    def run(lhs, lhs_scale, rhs, rhs_scale, output, stream=None):
        tensors = (lhs, lhs_scale, rhs, rhs_scale, output)
        pointers = tuple(
            make_ptr(
                dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
            )
            for dtype, align, tensor in zip(_TYPES, _ALIGN, tensors, strict=True)
        )
        run_compiled(
            raw,
            (*pointers, Int32(lhs.shape[0]), cuda_stream_from_int_or_current(stream)),
        )

    return attach_programs(run, raw)
