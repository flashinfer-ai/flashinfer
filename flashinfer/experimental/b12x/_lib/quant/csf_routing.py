"""Build an expert-presence mask once per compressed-scale invocation.

One warp scans the route vector for one expert. The bounded output is fully
overwritten on every call, including empty and invalid-only route vectors.
Scale decoders can share the result across all scale tiles without repeating
duplicate searches or expanding experts absent from the routed batch.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


class _ActiveExperts:
    @cute.jit
    def __call__(
        self,
        routes: cute.Pointer,
        mask: cute.Pointer,
        count: Int32,
        experts: Int32,
        stream: cuda.CUstream,
    ):
        ids = cute.make_tensor(routes, cute.make_layout((count,)))
        active = cute.make_tensor(mask, cute.make_layout((experts,)))
        self.kernel(ids, active, count, experts).launch(
            grid=((experts + Int32(7)) // Int32(8), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, ids, active, count: Int32, experts: Int32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        expert = Int32(block) * Int32(8) + Int32(tid) // Int32(32)
        lane = Int32(tid) % Int32(32)
        if expert < experts:
            found = cutlass.Boolean(False)
            route = lane
            while route < count:
                found = found | (ids[route].to(Int64) == expert.to(Int64))
                route += Int32(32)
            present = cute.arch.vote_any_sync(found)
            if lane == Int32(0):
                active[expert] = present.to(Int32)


@program_cache
def compile_csf_active_experts(ids64=False):
    kernel = _ActiveExperts()
    key = (bool(ids64),)
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    return b12x_compile(
        kernel,
        make_ptr(
            cutlass.Int64 if ids64 else cutlass.Int32,
            8 if ids64 else 4,
            cute.AddressSpace.gmem,
            assumed_align=8 if ids64 else 4,
        ),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.csf_active_experts", 1, key),
    )


def mark_active_experts(ids, active, program):
    import torch

    if (
        ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or ids.device != active.device
        or active.dtype != torch.int32
    ):
        raise ValueError("CSF presence masks require contiguous CUDA integer routes")
    program(
        make_ptr(
            cutlass.Int64 if ids.dtype == torch.int64 else cutlass.Int32,
            ids.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=8 if ids.dtype == torch.int64 else 4,
        ),
        make_ptr(
            cutlass.Int32, active.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        ),
        ids.numel(),
        active.numel(),
        current_cuda_stream(),
    )
