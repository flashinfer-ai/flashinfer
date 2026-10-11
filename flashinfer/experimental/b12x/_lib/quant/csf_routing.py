"""Build an expert-presence mask once per compressed-scale invocation.

One warp scans the route vector for one expert. The bounded output is fully
overwritten on every call, including empty and invalid-only route vectors.
Scale decoders can share the result across all scale tiles without repeating
duplicate searches or expanding experts absent from the routed batch.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.program_cache import program_cache, register_program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

_LAUNCH_PROGRAMS = {}
register_program_cache(_LAUNCH_PROGRAMS)


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


class _ActiveExpertBitmap:
    """Build per-warp bitmaps with work proportional to routes plus experts."""

    def __init__(self, experts):
        self.words = (experts + 31) // 32

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
            grid=(1, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, ids, active, count: Int32, experts: Int32):
        tid, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        bitmap = smem.allocate_tensor(
            cutlass.Uint32, cute.make_layout(8 * self.words), byte_alignment=16
        )
        index = Int32(tid)
        while index < Int32(8 * self.words):
            bitmap[index] = Uint32(0)
            index += Int32(256)
        cute.arch.sync_threads()
        route = Int32(tid)
        while route < count:
            expert = ids[route].to(Int64)
            if expert >= Int64(0) and expert < experts.to(Int64):
                word = Int32(tid) // Int32(32) * Int32(self.words) + expert.to(
                    Int32
                ) // Int32(32)
                bit = Uint32(1) << (expert.to(Uint32) % Uint32(32))
                cute.arch.atomic_or(
                    bitmap.iterator + word, bit, sem="relaxed", scope="cta"
                )
            route += Int32(256)
        cute.arch.sync_threads()
        expert = Int32(tid)
        while expert < experts:
            bits = Uint32(0)
            for warp in cutlass.range_constexpr(8):
                bits |= bitmap[Int32(warp * self.words) + expert // Int32(32)]
            active[expert] = ((bits >> (expert % Int32(32)).to(Uint32)) & Uint32(1)).to(
                Int32
            )
            expert += Int32(256)


@program_cache
def compile_csf_active_experts(ids64=False, experts=0):
    kernel = _ActiveExpertBitmap(experts) if experts else _ActiveExperts()
    key = (bool(ids64), int(experts))
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
        compile_spec=KernelCompileSpec.from_key("quant.csf_active_experts", 2, key),
    )


def mark_active_experts(ids, active, program):
    if (
        ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or ids.device != active.device
        or active.dtype != torch.int32
    ):
        raise ValueError("CSF presence masks require contiguous CUDA integer routes")
    handle = id(program)
    _LAUNCH_PROGRAMS[handle] = program
    _mark_active_experts_op(ids, active, handle)


@torch.library.custom_op("b12x::csf_active_experts", mutates_args=("active",))
def _mark_active_experts_op(
    ids: torch.Tensor, active: torch.Tensor, program_handle: int
) -> None:
    program = _LAUNCH_PROGRAMS[program_handle]
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


@_mark_active_experts_op.register_fake
def _mark_active_experts_fake(ids, active, program_handle) -> None:
    pass
