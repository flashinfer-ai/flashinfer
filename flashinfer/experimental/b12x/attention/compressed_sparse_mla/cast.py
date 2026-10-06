"""Prepared contiguous BF16/FP32 conversion into caller-owned storage."""
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int64

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan, make_fixed_contract
from b12x.preparation.types import plan_from_handle, require_prepared


@dataclass(frozen=True, kw_only=True)
class Query:
    max_elements: int
    input_dtype: str
    output_dtype: str

    def __post_init__(self):
        if type(self.max_elements) is not int or not 0 <= self.max_elements < 2**63:
            raise ValueError("cast capacity must be a nonnegative Int64 extent")
        if self.input_dtype not in ("bfloat16", "float32") or self.output_dtype not in ("bfloat16", "float32"):
            raise TypeError("cast requires BF16/FP32 input and output")


TUNING = make_fixed_contract(
    component_id="attention.compressed_sparse_mla.cast", query_type=Query, backend="cute",
)


class _Cast:
    def __init__(self, fp32):
        self.fp32 = fp32

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, size: Int64,
                 stream: cuda.CUstream):
        self.kernel(x, out, size).launch(grid=((size + Int64(255)) // Int64(256), 1, 1),
                                         block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, out: cute.Pointer, size: Int64):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        i = Int64(block) * Int64(256) + Int64(thread)
        if i < size:
            if cutlass.const_expr(self.fp32):
                out[i] = Float32(x[i])
            else:
                out[i] = BFloat16(x[i])



@dataclass(frozen=True)
class _CastProgram:
    raw: object
    types: tuple


@program_cache
def _compile_cast(input_dtype, output_dtype, device):
    key = (input_dtype, output_dtype, device)
    entry = _Cast(output_dtype == torch.float32)
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=key)
    types = tuple(BFloat16 if t == torch.bfloat16 else Float32
                  for t in (input_dtype, output_dtype))
    fake = tuple(make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8)
                 for t in types)
    with torch.cuda.device(device):
        raw = compile_cute(entry, *fake, Int64(1), current_cuda_stream(),
                           compile_spec=KernelCompileSpec.from_key(
                               "attention.compressed_sparse_mla.cast", 1, key))
    return attach_programs(_CastProgram(raw, types), raw)



@program_cache(scope="preparation")
def compile_cast(payload, ordinal):
    query = Query(**dict(payload))
    compiled = _compile_cast(getattr(torch, query.input_dtype), getattr(torch, query.output_dtype), ordinal)
    return attach_programs(_State(query, torch.device("cuda", ordinal), compiled.raw, compiled.types), compiled.raw)


@dataclass(frozen=True)
class _State:
    query: Query
    device: torch.device
    raw: object
    types: tuple

    def run(self, x, *, out):
        q = self.query
        if x.dtype != getattr(torch, q.input_dtype) or out.dtype != getattr(torch, q.output_dtype):
            raise TypeError("cast dtypes differ from the prepared conversion")
        if (x.shape != out.shape or not x.is_contiguous() or not out.is_contiguous()
                or x.device != self.device or out.device != self.device or x.numel() > q.max_elements):
            raise ValueError("cast requires matching contiguous tensors within its prepared capacity")
        if torch._C._overlaps(x, out) and (x.data_ptr() != out.data_ptr() or x.dtype != out.dtype):
            raise ValueError("cast input and output must not partially overlap")
        if x.numel():
            ptrs = tuple(make_ptr(t, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=t.width // 8)
                         for t, v in zip(self.types, (x, out), strict=True))
            with torch.cuda.device(self.device):
                run_compiled(self.raw, (*ptrs, Int64(x.numel()), current_cuda_stream()))
        return out


def plan(query: Query, *, device, invocation=FrozenMapping(), override=None):
    if invocation:
        raise ValueError("cast invocation is fully described by Query")

    def jobs(config, detected):
        return (CompileJob.create(
            "b12x.attention.compressed_sparse_mla.cast:compile_cast",
            TUNING.encode_query(query), detected.ordinal,
        ),)

    def materialize(selection, detected):
        state = compile_cast(TUNING.encode_query(query), detected.ordinal)
        load_programs(state)
        return state

    return Plan(contract=TUNING, query=query, override=override, _device=device, shared=True,
                _compile_jobs=jobs, _memory_requirements=lambda config, detected: MemoryRequirements(),
                _materialize=materialize)


@torch.library.custom_op("b12x::compressed_mla_cast", mutates_args=("out",))
def _cast_op(x: torch.Tensor, out: torch.Tensor, plan_handle: int) -> None:
    require_prepared(plan_from_handle(plan_handle), TUNING.component_id, x.device).run(x, out=out)


@_cast_op.register_fake
def _cast_fake(x: torch.Tensor, out: torch.Tensor, plan_handle: int) -> None:
    return None


def cast(x, *, out, plan: Plan):
    """Convert live elements through the prepared launcher into caller storage."""
    _cast_op(x, out, plan.handle)
    return out
