"""Prepared BF16 rounding boundary for V4.1 index-head weights."""
from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan, make_fixed_contract
from b12x.preparation.types import require_prepared


@dataclass(frozen=True, kw_only=True)
class Query:
    max_elements: int

    def __post_init__(self):
        if type(self.max_elements) is not int or not 0 < self.max_elements < 2**31:
            raise ValueError("index-weight capacity must be a positive Int32 element count")


TUNING = make_fixed_contract(
    component_id="attention.compressed_sparse_mla.index_weights", query_type=Query, backend="cute",
)


class _Scale:
    @cute.jit
    def __call__(self, x: cute.Pointer, y: cute.Pointer, n: Int32,
                 stream: cuda.CUstream):
        self.kernel(x, y, n).launch(grid=((n + 255) // 256, 1, 1),
                                     block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, y: cute.Pointer, n: Int32):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        i = block * 256 + thread
        if i < n:
            y[Int64(i)] = BFloat16(Float32(x[Int64(i)]) * Float32(1.0 / 64.0))


@program_cache(scope="preparation")
def compile_scale(ordinal):
    pointer = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2)
    with torch.cuda.device(ordinal):
        return compile_cute(
            _Scale(), pointer, pointer, Int32(1), current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "attention.compressed_sparse_mla.index_weights", 1, (ordinal,),
            ),
        )


@dataclass(frozen=True)
class _State:
    query: Query
    device: torch.device
    program: object

    def run(self, x, *, out):
        if x.dtype != torch.bfloat16 or out.dtype != x.dtype or out.shape != x.shape:
            raise ValueError("index weights require matching BF16 tensors")
        if (not x.is_contiguous() or not out.is_contiguous()
                or x.device != self.device or out.device != self.device
                or x.numel() > self.query.max_elements):
            raise ValueError("index weights exceed the prepared device/layout/capacity")
        if x.numel():
            pointers = tuple(make_ptr(BFloat16, t.data_ptr(), cute.AddressSpace.gmem,
                                     assumed_align=2) for t in (x, out))
            with torch.cuda.device(self.device):
                run_compiled(self.program, (*pointers, Int32(x.numel()), current_cuda_stream()))
        return out


def plan(query: Query, *, device, invocation=FrozenMapping(), override=None) -> Plan:
    if invocation:
        raise ValueError("index-weight invocation is fully described by Query")

    def jobs(config, detected):
        return (CompileJob.create(
            "b12x.attention.compressed_sparse_mla.weight_scale:compile_scale", detected.ordinal,
        ),)

    def materialize(selection, detected):
        program = compile_scale(detected.ordinal)
        load_programs(program)
        return attach_programs(_State(query, torch.device("cuda", detected.ordinal), program), program)

    return Plan(contract=TUNING, query=query, override=override, _device=device,
                _compile_jobs=jobs, _memory_requirements=lambda config, detected: MemoryRequirements(),
                _materialize=materialize)


def scale_index_weights(x, *, out, plan):
    return require_prepared(plan, TUNING.component_id, x.device).run(x, out=out)
