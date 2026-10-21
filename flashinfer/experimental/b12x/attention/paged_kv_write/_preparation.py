"""Metadata compilation and retained launch for paged K/V writes."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_tensor

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.compiler import KernelCompileSpec, run_compiled
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan

from ._tuning import PagedKVWriteQuery, TUNING


def _operands(fp8):
    return (
        ("key", 3, cutlass.BFloat16, 16),
        ("value", 3, cutlass.BFloat16, 16),
        ("k_cache", 4, cutlass.Uint8 if fp8 else cutlass.BFloat16, 16),
        ("v_cache", 4, cutlass.Uint8 if fp8 else cutlass.BFloat16, 16),
        ("slot_mapping", 1, cutlass.Int64, 8),
        ("k_scale", 1, cutlass.Float32, 4),
        ("v_scale", 1, cutlass.Float32, 4),
    )


@program_cache(scope="preparation")
def compile_writer(payload, ordinal):
    from ..paged_decode._kv_write import PagedKVWriteKernel

    q = PagedKVWriteQuery(**dict(payload))
    fp8 = q.kv_dtype == "float8_e4m3fn"
    operands = tuple(
        make_fake_tensor(
            dtype,
            tuple(cute.sym_int(32) for _ in range(rank)),
            tuple(cute.sym_int(64) for _ in range(rank - 1)) + (1,),
            assumed_align=align,
        )
        for _, rank, dtype, align in _operands(fp8)
    )
    with torch.cuda.device(ordinal):
        return b12x_compile(
            PagedKVWriteKernel(
                num_kv_heads=q.num_kv_heads,
                head_dim_k=q.head_dim_k,
                head_dim_v=q.head_dim_v,
                page_size=q.page_size,
                kv_fp8=fp8,
            ),
            *operands,
            cutlass.Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "attention.paged_kv_write",
                1,
                (q.num_kv_heads, q.head_dim_k, q.head_dim_v, q.page_size, q.kv_dtype),
            ),
        )


def _check_layout(name, tensor, *, device, rank, align):
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if tensor.ndim != rank or tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have rank {rank} and unit inner stride")
    if any(size >= 2**31 for size in tensor.shape):
        raise ValueError(f"{name} shape exceeds the Int32 indexing domain")
    if tensor.numel() and tensor.data_ptr() % align:
        raise ValueError(f"{name} must be {align}-byte aligned")
    if any(stride * tensor.element_size() % align for stride in tensor.stride()[:-1]):
        raise ValueError(f"{name} outer strides must be {align}-byte aligned")
    span = 1
    for size, stride in sorted(zip(tensor.shape, tensor.stride()), key=lambda x: x[1]):
        if size > 1:
            if stride < span:
                raise ValueError(f"{name} must have nonoverlapping positive strides")
            span += (size - 1) * stride


@dataclass(frozen=True)
class PreparedPagedKVWrite:
    query: PagedKVWriteQuery
    device: torch.device
    program: object
    dummy_scale: torch.Tensor

    def run(
        self, *, key, value, k_cache, v_cache, slot_mapping, k_scale=None, v_scale=None
    ):
        q = self.query
        fp8 = q.kv_dtype == "float8_e4m3fn"
        tensors = dict(
            key=key,
            value=value,
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mapping,
        )
        for name, rank, _, align in _operands(fp8)[:5]:
            _check_layout(
                name, tensors[name], device=self.device, rank=rank, align=align
            )
        tokens = slot_mapping.numel()
        if slot_mapping.dtype != torch.int64:
            raise ValueError("slot_mapping must be contiguous int64 [T]")
        for name, rows, dim in (
            ("key", key, q.head_dim_k),
            ("value", value, q.head_dim_v),
        ):
            if rows.dtype != torch.bfloat16 or tuple(rows.shape[1:]) != (
                q.num_kv_heads,
                dim,
            ):
                raise ValueError(f"{name} must be BF16 [T, {q.num_kv_heads}, {dim}]")
            if rows.shape[0] < tokens:
                raise ValueError(f"{name} has fewer rows than slot_mapping")
        allowed = (torch.float8_e4m3fn, torch.uint8) if fp8 else (torch.bfloat16,)
        for name, cache, dim in (
            ("k_cache", k_cache, q.head_dim_k),
            ("v_cache", v_cache, q.head_dim_v),
        ):
            if tuple(cache.shape[1:]) != (q.page_size, q.num_kv_heads, dim):
                raise ValueError(f"{name} geometry differs from preparation")
            if cache.dtype not in allowed:
                raise ValueError(f"{name} dtype differs from preparation")
        if k_cache.shape[0] != v_cache.shape[0]:
            raise ValueError("K/V caches must have the same page count")
        if fp8:
            for name, scale in (("k_scale", k_scale), ("v_scale", v_scale)):
                if scale is None or scale.dtype != torch.float32 or scale.numel() != 1:
                    raise ValueError(f"{name} must be one float32 value")
                scale = scale.reshape(1)
                _check_layout(name, scale, device=self.device, rank=1, align=4)
                tensors[name] = scale
            tensors["k_cache"] = k_cache.view(torch.uint8)
            tensors["v_cache"] = v_cache.view(torch.uint8)
        else:
            tensors["k_scale"] = tensors["v_scale"] = self.dummy_scale
        if not tokens:
            return
        args = []
        for name, _, dtype, align in _operands(fp8):
            tensor = tensors[name]
            converted = from_dlpack(tensor, assumed_align=align)
            converted.element_type = dtype
            args.append(converted.mark_layout_dynamic(leading_dim=tensor.ndim - 1))
        with torch.cuda.device(self.device):
            run_compiled(
                self.program, (*args, cutlass.Int32(tokens), current_cuda_stream())
            )


def make_plan(caps, *, invocation=None, override=None):
    from .api import Caps

    if not isinstance(caps, Caps):
        raise TypeError("caps must be paged_kv_write.Caps")
    if invocation:
        raise ValueError("paged KV write declarations carry no invocation metadata")
    query = PagedKVWriteQuery(
        num_kv_heads=caps.num_kv_heads,
        head_dim_k=caps.head_dim_k,
        head_dim_v=caps.head_dim_v,
        page_size=caps.page_size,
        kv_dtype=str(caps.kv_dtype).removeprefix("torch."),
    )
    device = torch.device(caps.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    def jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.attention.paged_kv_write._preparation:compile_writer",
                TUNING.encode_query(query),
                detected.ordinal,
            ),
        )

    def materialize(selection, detected):
        program = compile_writer(TUNING.encode_query(query), detected.ordinal)
        load_programs(program)
        device = torch.device("cuda", detected.ordinal)
        return attach_programs(
            PreparedPagedKVWrite(
                query,
                device,
                program,
                torch.ones(1, dtype=torch.float32, device=device),
            ),
            program,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(),
        override=override,
        _device=device,
        _compile_jobs=jobs,
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=materialize,
    )
