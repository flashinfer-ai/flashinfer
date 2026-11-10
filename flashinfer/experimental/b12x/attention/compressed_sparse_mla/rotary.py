"""Prepared V4.1 BF16 rotary transform, without per-query-head RMSNorm."""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    make_fixed_contract,
)
from b12x.preparation.types import require_prepared


@dataclass(frozen=True, kw_only=True)
class Query:
    max_rows: int
    heads: int
    dim: int
    rope_dim: int = 64
    inverse: bool = False
    ratio: int = 1
    cos_sin_dtype: str = "float32"

    def __post_init__(self):
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.max_rows,
                self.heads,
                self.dim,
                self.rope_dim,
                self.ratio,
            )
        ):
            raise ValueError("rotary dimensions and capacity must be positive integers")
        if self.max_rows >= 2**31 or self.rope_dim > self.dim or self.rope_dim % 2:
            raise ValueError(
                "rotary requires Int32 row capacity and an even bounded rope dimension"
            )
        if type(self.inverse) is not bool or self.cos_sin_dtype not in (
            "bfloat16",
            "float32",
        ):
            raise ValueError("rotary requires a boolean direction and BF16/FP32 table")


TUNING = make_fixed_contract(
    component_id="attention.compressed_sparse_mla.rotate",
    query_type=Query,
    backend="cute",
)


class _Rotate:
    def __init__(self, heads, dim, rope_dim, inverse, ratio):
        self.heads, self.dim, self.rope_dim = heads, dim, rope_dim
        self.inverse, self.ratio = inverse, ratio

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        pos: cute.Pointer,
        cs: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        sx: Int64,
        sh: Int64,
        sc: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(x, pos, cs, out, rows, sx, sh, sc).launch(
            grid=(rows, self.heads, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Pointer,
        pos: cute.Pointer,
        cs: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        sx: Int64,
        sh: Int64,
        sc: Int64,
    ):
        row, head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        position = Int64(pos[row])
        if cutlass.const_expr(self.ratio > 1):
            position = (position // Int64(self.ratio)) * Int64(self.ratio)
        for item in cutlass.range_constexpr((self.dim + 127) // 128):
            col = tid + item * 128
            if col < self.dim:
                offset = Int64(row) * sx + Int64(head) * sh + Int64(col)
                value = Float32(0.0)
                if position >= 0:
                    value = Float32(x[offset])
                    if col >= self.dim - self.rope_dim:
                        local = col - (self.dim - self.rope_dim)
                        partner = Float32(x[offset + Int64(1 - 2 * (local % 2))])
                        cosine = Float32(cs[position * sc + Int64(local // 2)])
                        sine = Float32(
                            cs[position * sc + Int64(self.rope_dim // 2 + local // 2)]
                        )
                        sign = Float32(-1.0)
                        if local % 2 == 1:
                            sign = Float32(1.0)
                        if cutlass.const_expr(self.inverse):
                            sign = -sign
                        value = value * cosine + sign * partner * sine
                out[
                    (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(self.dim)
                    + Int64(col)
                ] = BFloat16(value)


@program_cache(scope="preparation")
def compile_rotation(payload, ordinal):
    query = Query(**dict(payload))
    cs_dtype = getattr(torch, query.cos_sin_dtype)
    key = (
        query.heads,
        query.dim,
        query.rope_dim,
        query.inverse,
        query.ratio,
        cs_dtype,
        ordinal,
    )
    entry = _Rotate(query.heads, query.dim, query.rope_dim, query.inverse, query.ratio)
    types = (
        BFloat16,
        Int64,
        Float32 if cs_dtype == torch.float32 else BFloat16,
        BFloat16,
    )
    pointers = tuple(
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8)
        for t in types
    )
    with torch.cuda.device(ordinal):
        return compile_cute(
            entry,
            *pointers,
            Int32(1),
            Int64(1),
            Int64(1),
            Int64(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "attention.compressed_sparse_mla.rotate", 1, key
            ),
        )


@dataclass(frozen=True)
class _State:
    query: Query
    device: torch.device
    program: object

    def run(self, x, positions, cos_sin_cache, *, out):
        q = self.query
        if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise TypeError("rotary input/output must be BF16")
        heads = x.shape[1] if x.ndim == 3 else 1
        if (
            x.ndim not in (2, 3)
            or heads != q.heads
            or x.shape[-1] != q.dim
            or x.shape[0] > q.max_rows
            or x.stride(-1) != 1
            or out.shape != x.shape
            or not out.is_contiguous()
        ):
            raise ValueError("rotary tensors differ from the prepared geometry")
        if (
            positions.dtype != torch.int64
            or positions.shape != (x.shape[0],)
            or not positions.is_contiguous()
        ):
            raise ValueError("rotary positions must be contiguous int64[T]")
        if (
            cos_sin_cache.dtype != getattr(torch, q.cos_sin_dtype)
            or cos_sin_cache.ndim != 2
            or cos_sin_cache.shape[1] < q.rope_dim
            or cos_sin_cache.stride(1) != 1
        ):
            raise ValueError("rotary table differs from the prepared dtype/layout")
        if any(t.device != self.device for t in (x, positions, cos_sin_cache, out)):
            raise ValueError("rotary tensors must share the prepared CUDA device")
        if x.data_ptr() == out.data_ptr():
            raise ValueError("rotary output must not alias its input")
        if x.shape[0]:
            types = (
                BFloat16,
                Int64,
                Float32 if q.cos_sin_dtype == "float32" else BFloat16,
                BFloat16,
            )
            args = tuple(
                make_ptr(
                    t, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=t.width // 8
                )
                for t, v in zip(types, (x, positions, cos_sin_cache, out), strict=True)
            )
            with torch.cuda.device(self.device):
                run_compiled(
                    self.program,
                    (
                        *args,
                        Int32(x.shape[0]),
                        Int64(x.stride(0)),
                        Int64(x.stride(1) if x.ndim == 3 else 0),
                        Int64(cos_sin_cache.stride(0)),
                        current_cuda_stream(),
                    ),
                )
        return out


def plan(query: Query, *, device, invocation=FrozenMapping(), override=None) -> Plan:
    if invocation:
        raise ValueError("rotary invocation is fully described by Query")

    def jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.attention.compressed_sparse_mla.rotary:compile_rotation",
                TUNING.encode_query(query),
                detected.ordinal,
            ),
        )

    def materialize(selection, detected):
        program = compile_rotation(TUNING.encode_query(query), detected.ordinal)
        load_programs(program)
        return attach_programs(
            _State(query, torch.device("cuda", detected.ordinal), program), program
        )

    return Plan(
        contract=TUNING,
        query=query,
        override=override,
        _device=device,
        _compile_jobs=jobs,
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=materialize,
    )


def rotate(x, positions, cos_sin_cache, *, out, plan):
    """Rotate into caller storage using the admitted direction and group ratio."""
    return require_prepared(plan, TUNING.component_id, x.device).run(
        x,
        positions,
        cos_sin_cache,
        out=out,
    )
