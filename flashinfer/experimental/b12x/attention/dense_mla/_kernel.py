"""Compilation and graph-safe launch plumbing for native dense MLA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    compile as compile_cute,
    key_field,
    run_compiled,
    tensor_key,
)
from b12x._lib.compile_plan import compile_only_launches_enabled

from .._shared import static_fp8_quant
from ._forward import DenseMlaForwardKernel
from ._layout import make_smem_layout
from ._merge import DenseMlaMergeKernel
from ._scratch import Binding
from b12x._lib.program_cache import register_program_cache

_FP8 = torch.float8_e4m3fn
_LOG2_E = math.log2(math.e)
_LOCK = RLock()
_FORWARD_CACHE: dict[tuple[object, ...], object] = {}
_MERGE_CACHE: dict[tuple[object, ...], object] = {}
register_program_cache(_FORWARD_CACHE, lock=_LOCK)
register_program_cache(_MERGE_CACHE, lock=_LOCK)
def _to_cute(
    tensor: torch.Tensor,
    dtype,
    *,
    align: int,
    dynamic_layout: bool = False,
):
    if compile_only_launches_enabled() and hasattr(tensor, "fake_mode"):
        from cutlass.cute.runtime import make_fake_tensor
        leading_dim = next(
            (index for index, stride in enumerate(tensor.stride()) if int(stride) == 1),
            None,
        )
        if dynamic_layout and tensor.ndim and leading_dim is not None:
            shape = tuple(cutlass.cute.sym_int(32) for _ in tensor.shape)
            strides = tuple(
                1 if index == leading_dim else cutlass.cute.sym_int(64)
                for index in range(tensor.ndim)
            )
        else:
            shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        return make_fake_tensor(dtype, shape, strides, assumed_align=align)
    converted = from_dlpack(tensor, assumed_align=align)
    converted.element_type = dtype
    if dynamic_layout and tensor.ndim:
        leading_dim = next(
            (index for index, stride in enumerate(tensor.stride()) if int(stride) == 1),
            None,
        )
        if leading_dim is not None:
            converted = converted.mark_layout_dynamic(leading_dim=leading_dim)
    return converted


def _byte_base_pointer(tensor: torch.Tensor) -> torch.Tensor:
    """Return a one-element byte view at the tensor's physical base.

    Kernel addressing is entirely explicit and Int64. Exposing the full pool
    span through DLPack would force CUTLASS's memref descriptor to encode a
    length beyond signed Int32 for the exact high-pid serving pools this ABI is
    required to support.
    """
    byte_view = tensor.view(torch.uint8)
    return torch.as_strided(byte_view, size=(1,), stride=(1,))


def _signature(binding: Binding) -> tuple[object, ...]:
    scratch = binding.scratch
    device = binding.q.device
    device_index = (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )
    return (
        device_index,
        str(binding.q.dtype),
        scratch.num_q_heads,
        scratch.page_size,
        scratch.max_total_q,
        scratch.max_batch,
        scratch.query_tile,
        scratch.num_splits,
        scratch.chunks_per_split,
        scratch.physical_record_width,
        scratch.head_dim,
        scratch.v_head_dim,
        scratch.window_size,
        tuple(int(value) for value in binding.q.stride()),
        tuple(int(value) for value in binding.kv_cache.stride()),
        tuple(int(value) for value in binding.output.stride()),
        (
            tuple(int(value) for value in scratch.partial_output.stride())
            if scratch.partial_output is not None
            else ()
        ),
    )


@dataclass(frozen=True)
class _ForwardLaunch:
    entry: DenseMlaForwardKernel
    args: tuple[object, ...]
    spec: KernelCompileSpec


@dataclass(frozen=True)
class _MergeLaunch:
    entry: DenseMlaMergeKernel
    args: tuple[object, ...]
    spec: KernelCompileSpec


def _forward_launch(binding: Binding) -> _ForwardLaunch:
    scratch = binding.scratch
    fp8 = binding.q.dtype == _FP8
    layout = make_smem_layout(
        query_tile=scratch.query_tile,
        fp8=fp8,
        qk_dim=scratch.head_dim,
    )
    entry = DenseMlaForwardKernel(
        layout=layout,
        page_size=scratch.page_size,
        num_heads=scratch.num_q_heads,
        num_splits=scratch.num_splits,
        chunks_per_split=scratch.chunks_per_split,
        query_tile=scratch.query_tile,
        fp8=fp8,
        qk_dim=scratch.head_dim,
        value_dim=scratch.v_head_dim,
        window_size=scratch.window_size,
    )

    q_bytes = _byte_base_pointer(binding.q)
    cache_bytes = _byte_base_pointer(binding.kv_cache)
    page_table = binding.page_table.reshape(-1)
    output = binding.output
    final_lse = scratch.final_lse
    partial_output = (
        scratch.partial_output if scratch.partial_output is not None else output
    )
    partial_lse = scratch.partial_lse if scratch.partial_lse is not None else final_lse
    unused_scale_storage = final_lse.reshape(-1)[:1]
    kv_scale = (
        binding.kv_scale if binding.kv_scale is not None else unused_scale_storage
    )
    q_scale = binding.q_scale if binding.q_scale is not None else unused_scale_storage
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute(
            q_bytes,
            cutlass.Uint8,
            align=16,
            dynamic_layout=True,
        ),
        _to_cute(
            cache_bytes,
            cutlass.Uint8,
            align=16,
            dynamic_layout=True,
        ),
        _to_cute(
            page_table,
            cutlass.Int32,
            align=4,
            dynamic_layout=True,
        ),
        _to_cute(
            binding.cache_seqlens,
            cutlass.Int32,
            align=4,
            dynamic_layout=True,
        ),
        _to_cute(
            binding.cu_seqlens_q,
            cutlass.Int32,
            align=4,
            dynamic_layout=True,
        ),
        _to_cute(
            output,
            cutlass.BFloat16,
            align=16,
            dynamic_layout=True,
        ),
        _to_cute(final_lse, cutlass.Float32, align=4),
        _to_cute(partial_output, cutlass.BFloat16, align=16),
        _to_cute(partial_lse, cutlass.Float32, align=4),
        _to_cute(kv_scale, cutlass.Float32, align=4),
        _to_cute(q_scale, cutlass.Float32, align=4),
        Float32(float(binding.sm_scale) * _LOG2_E),
        Int64(int(binding.q.stride(0)) * binding.q.element_size()),
        Int64(int(binding.q.stride(1)) * binding.q.element_size()),
        Int64(int(binding.kv_cache.stride(0)) * binding.kv_cache.element_size()),
        Int64(int(binding.kv_cache.stride(1)) * binding.kv_cache.element_size()),
        Int64(int(binding.page_table.stride(0))),
        Int32(int(binding.q.shape[0])),
        Int32(int(binding.cache_seqlens.shape[0])),
        Int32(binding.active_splits),
        stream,
    )
    spec = KernelCompileSpec.from_fields(
        "attention.dense_mla.forward",
        4,
        key_field("dtype", "fp8" if fp8 else "bf16"),
        key_field("heads", scratch.num_q_heads),
        key_field("page_size", scratch.page_size),
        key_field("max_total_q", scratch.max_total_q),
        key_field("max_batch", scratch.max_batch),
        key_field("query_tile", scratch.query_tile),
        key_field("num_splits", scratch.num_splits),
        key_field("chunks_per_split", scratch.chunks_per_split),
        key_field("physical_record_width", scratch.physical_record_width),
        key_field("head_dim", scratch.head_dim),
        key_field("v_head_dim", scratch.v_head_dim),
        key_field("window_size", scratch.window_size),
        key_field("record_stride_bytes", layout.record_stride_bytes),
        tensor_key(
            "q",
            binding.q,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(scratch.num_q_heads),
                DimKey.exact(scratch.head_dim),
            ),
        ),
        tensor_key(
            "cache",
            binding.kv_cache,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(scratch.page_size),
                DimKey.exact(scratch.physical_record_width),
            ),
        ),
        tensor_key(
            "output",
            output,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(scratch.num_q_heads),
                DimKey.exact(scratch.v_head_dim),
            ),
        ),
    )
    return _ForwardLaunch(entry=entry, args=args, spec=spec)


def _merge_launch(binding: Binding) -> _MergeLaunch | None:
    scratch = binding.scratch
    if scratch.num_splits == 1:
        return None
    assert scratch.partial_output is not None
    assert scratch.partial_lse is not None
    entry = DenseMlaMergeKernel(scratch.num_splits, scratch.v_head_dim)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute(
            scratch.partial_output,
            cutlass.BFloat16,
            align=16,
        ),
        _to_cute(scratch.partial_lse, cutlass.Float32, align=4),
        _to_cute(
            binding.output,
            cutlass.BFloat16,
            align=16,
            dynamic_layout=True,
        ),
        _to_cute(scratch.final_lse, cutlass.Float32, align=4),
        Int32(binding.active_splits),
        stream,
    )
    spec = KernelCompileSpec.from_fields(
        "attention.dense_mla.merge",
        4,
        key_field("num_splits", scratch.num_splits),
        key_field("heads", scratch.num_q_heads),
        key_field("v_head_dim", scratch.v_head_dim),
        tensor_key(
            "partial_output",
            scratch.partial_output,
            dims=(
                DimKey.capacity(scratch.max_total_q),
                DimKey.exact(scratch.num_q_heads),
                DimKey.exact(scratch.num_splits),
                DimKey.exact(scratch.v_head_dim),
            ),
        ),
        tensor_key(
            "output",
            binding.output,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(scratch.num_q_heads),
                DimKey.exact(scratch.v_head_dim),
            ),
        ),
    )
    return _MergeLaunch(entry=entry, args=args, spec=spec)


def _compile_entries(binding: Binding) -> tuple[object, object | None]:
    signature = _signature(binding)
    with _LOCK:
        compiled_forward = _FORWARD_CACHE.get(signature)
        compiled_merge = _MERGE_CACHE.get(signature)
    if compiled_forward is None:
        launch = _forward_launch(binding)
        compiled_forward = compile_cute(
            launch.entry,
            *launch.args,
            compile_spec=launch.spec,
        )
        with _LOCK:
            _FORWARD_CACHE[signature] = compiled_forward
    if binding.scratch.num_splits > 1 and compiled_merge is None:
        launch = _merge_launch(binding)
        assert launch is not None
        compiled_merge = compile_cute(
            launch.entry,
            *launch.args,
            compile_spec=launch.spec,
        )
        with _LOCK:
            _MERGE_CACHE[signature] = compiled_merge
    return compiled_forward, compiled_merge

@dataclass(frozen=True)
class DenseMlaLaunchers:
    """Resolved native entries retained by a prepared sparse-MLA state."""

    signature: tuple[object, ...]
    forward: object
    merge: object | None

    def run(self, binding: Binding) -> tuple[torch.Tensor, torch.Tensor]:
        if _signature(binding) != self.signature:
            raise ValueError("dense MLA binding differs from its prepared launcher")
        forward = _forward_launch(binding)
        run_compiled(self.forward, forward.args)
        if binding.scratch.num_splits > 1 and binding.active_splits > 1:
            if self.merge is None:
                raise RuntimeError("prepared dense MLA binding is missing its merge launcher")
            merge = _merge_launch(binding)
            assert merge is not None
            run_compiled(self.merge, merge.args)
        rows = int(binding.q.shape[0])
        return binding.output, binding.scratch.final_lse[:rows]


def resolve_dense_mla_launchers(*, binding: Binding) -> DenseMlaLaunchers:
    """Resolve and retain the existing exact forward/merge native entries."""
    if not binding.q.is_cuda:
        raise ValueError("dense MLA native compilation requires CUDA tensors")
    forward, merge = _compile_entries(binding)
    return DenseMlaLaunchers(_signature(binding), forward, merge)


def compile_dense_mla(*, binding: Binding) -> None:
    """Compile the exact native forward/merge entries without launching."""
    if not binding.q.is_cuda:
        raise ValueError("dense MLA native compilation requires CUDA tensors")
    if binding.query_quant is not None:
        static_fp8_quant.compile(binding=binding.query_quant)
    _compile_entries(binding)


def run_dense_mla(
    *,
    binding: Binding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch only entries resolved during preparation."""
    if not binding.q.is_cuda:
        raise ValueError("dense MLA native execution requires CUDA tensors")
    if binding.query_quant is not None:
        static_fp8_quant.run(binding=binding.query_quant)
    signature = _signature(binding)
    with _LOCK:
        compiled_forward = _FORWARD_CACHE.get(signature)
        compiled_merge = _MERGE_CACHE.get(signature)
    if compiled_forward is None or (
        binding.scratch.num_splits > 1 and compiled_merge is None
    ):
        raise RuntimeError(
            "dense MLA execution is not prepared; resolve launchers before run"
        )

    forward = _forward_launch(binding)
    run_compiled(compiled_forward, forward.args)
    if binding.scratch.num_splits > 1 and binding.active_splits > 1:
        assert compiled_merge is not None
        merge = _merge_launch(binding)
        assert merge is not None
        run_compiled(compiled_merge, merge.args)
    rows = int(binding.q.shape[0])
    return binding.output, binding.scratch.final_lse[:rows]


def clear_dense_mla_kernel_caches() -> None:
    static_fp8_quant.clear_caches()
    with _LOCK:
        _FORWARD_CACHE.clear()
        _MERGE_CACHE.clear()


__all__ = [
    "DenseMlaLaunchers",
    "clear_dense_mla_kernel_caches",
    "compile_dense_mla",
    "resolve_dense_mla_launchers",
    "run_dense_mla",
]
