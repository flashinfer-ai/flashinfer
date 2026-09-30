"""Partial summation for BF16-activation weight-only GEMM."""
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cuda.bindings import driver as cuda

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen


class WeightOnlySplitKReduce:
    def __init__(self, n: int, slices: int):
        self.n = n
        self.slices = slices

    @cute.jit
    def __call__(self, partials: cute.Pointer, output: cute.Pointer,
                 m: Int32, stream: cuda.CUstream):
        self.kernel(partials, output, m).launch(
            grid=((Int64(m) * self.n + 255) // 256, 1, 1),
            block=(256, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, partials: cute.Pointer, output: cute.Pointer, m: Int32):
        offset = Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        size = Int64(m) * self.n
        if offset < size:
            value = cutlass.Float32(0)
            for part in cutlass.range_constexpr(self.slices):
                value += partials[Int64(part) * size + offset]
            output[offset] = cutlass.BFloat16(value)


@program_cache
def compile_reduce(n: int, slices: int, device: int):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("A16 split-K reduction must be prewarmed before CUDA graph capture")
    launch = WeightOnlySplitKReduce(n, slices)
    key = (n, slices, device)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        1, current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("gemm.dense.split_k_reduce", 1, key),
    )
