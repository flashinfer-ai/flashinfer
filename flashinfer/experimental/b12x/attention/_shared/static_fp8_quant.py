"""Graph-safe BF16 to per-tensor E4M3 query quantization."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.compiler import key_field, run_compiled
from b12x._lib.program_cache import register_program_cache
from b12x._lib.intrinsics import (
    bfloat2_to_float2_scaled,
    cvt_f32x4_to_e4m3x4,
    get_ptr_as_int64,
    ld_global_v2_u32,
    st_global_u32,
)

FP8 = torch.float8_e4m3fn
_THREADS = 128
_VALUES_PER_THREAD = 4
_LOCK = RLock()
_CACHE: dict[tuple[int, int], object] = {}
register_program_cache(_CACHE, lock=_LOCK)


def _byte_base_pointer(tensor: torch.Tensor) -> torch.Tensor:
    byte_view = tensor.view(torch.uint8)
    return torch.as_strided(byte_view, size=(1,), stride=(1,))


def _to_cute(tensor: torch.Tensor, dtype, *, align: int):
    if hasattr(tensor, "fake_mode"):
        from cutlass.cute.runtime import make_fake_tensor
        return make_fake_tensor(
            dtype, (cute.sym_int(32),), (1,), assumed_align=align
        )
    converted = from_dlpack(tensor, assumed_align=align)
    converted.element_type = dtype
    return converted.mark_layout_dynamic(leading_dim=0)


class _StaticFp8QuantKernel:
    def __init__(self, max_numel: int):
        self.max_numel = int(max_numel)

    @cute.jit
    def __call__(
        self,
        source_bytes: cute.Tensor,
        output_bytes: cute.Tensor,
        scale: cute.Tensor,
        active_numel: Int32,
        stream: cuda.CUstream,
    ):
        grid = (self.max_numel + _THREADS * _VALUES_PER_THREAD - 1) // (
            _THREADS * _VALUES_PER_THREAD
        )
        self.kernel(source_bytes, output_bytes, scale, active_numel).launch(
            grid=(grid, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source_bytes: cute.Tensor,
        output_bytes: cute.Tensor,
        scale: cute.Tensor,
        active_numel: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        word = Int32(block) * Int32(_THREADS) + Int32(thread)
        element = word * Int32(_VALUES_PER_THREAD)
        if element < active_numel:
            inverse_scale = Float32(1.0) / Float32(scale[0])
            lo, hi = ld_global_v2_u32(
                get_ptr_as_int64(source_bytes, element.to(Int64) * Int64(2))
            )
            value0, value1 = bfloat2_to_float2_scaled(lo, inverse_scale)
            value2, value3 = bfloat2_to_float2_scaled(hi, inverse_scale)
            packed = cvt_f32x4_to_e4m3x4(value0, value1, value2, value3)
            st_global_u32(
                get_ptr_as_int64(output_bytes, element.to(Int64)),
                packed,
            )


@dataclass(frozen=True)
class Binding:
    source: torch.Tensor
    output: torch.Tensor
    scale: torch.Tensor
    max_numel: int


def bind(
    *,
    source: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
    max_numel: int,
) -> Binding:
    if source.dtype != torch.bfloat16 or not source.is_contiguous():
        raise TypeError("query source must be contiguous BF16")
    if output.dtype != FP8 or tuple(output.shape) != tuple(source.shape):
        raise TypeError("query quant output must be E4M3 with the source shape")
    if not output.is_contiguous() or output.device != source.device:
        raise ValueError("query quant output must be contiguous on the source device")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise TypeError("query quant scale must be a scalar float32 tensor")
    if scale.device != source.device:
        raise ValueError("query quant scale must be on the source device")
    if source.numel() % _VALUES_PER_THREAD:
        raise ValueError("query element count must be divisible by four")
    if not 0 < source.numel() <= int(max_numel):
        raise ValueError("query element count exceeds the planned quant capacity")
    return Binding(
        source=source.detach(),
        output=output.detach(),
        scale=scale.detach(),
        max_numel=int(max_numel),
    )


def _launch(binding: Binding):
    source_bytes = _byte_base_pointer(binding.source)
    output_bytes = _byte_base_pointer(binding.output)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    entry = _StaticFp8QuantKernel(binding.max_numel)
    args = (
        _to_cute(source_bytes, cutlass.Uint8, align=16),
        _to_cute(output_bytes, cutlass.Uint8, align=16),
        _to_cute(binding.scale.reshape(1), cutlass.Float32, align=4),
        Int32(binding.source.numel()),
        stream,
    )
    spec = KernelCompileSpec.from_fields(
        "attention.static_fp8_query_quant",
        1,
        key_field("max_numel", binding.max_numel),
    )
    return entry, args, spec


def _signature(binding: Binding) -> tuple[int, int]:
    index = binding.source.device.index
    if index is None:
        index = torch.cuda.current_device()
    return int(index), int(binding.max_numel)


def compile(*, binding: Binding) -> None:
    signature = _signature(binding)
    with _LOCK:
        compiled = _CACHE.get(signature)
    if compiled is None:
        entry, args, spec = _launch(binding)
        compiled = compile_cute(entry, *args, compile_spec=spec)
        with _LOCK:
            _CACHE[signature] = compiled

@dataclass(frozen=True)
class StaticFp8QuantLauncher:
    signature: tuple[int, int]
    compiled: object

    def run(self, binding: Binding) -> torch.Tensor:
        if _signature(binding) != self.signature:
            raise ValueError("static FP8 quant binding differs from its prepared launcher")
        _, args, _ = _launch(binding)
        run_compiled(self.compiled, args)
        return binding.output


def resolve_static_fp8_quant_launcher(*, binding: Binding) -> StaticFp8QuantLauncher:
    """Resolve the exact native quantizer and retain it for prepared execution."""
    compile(binding=binding)
    signature = _signature(binding)
    with _LOCK:
        compiled = _CACHE[signature]
    return StaticFp8QuantLauncher(signature, compiled)


def run(*, binding: Binding) -> torch.Tensor:
    """Run a launcher resolved during preparation; never compile at execution."""
    signature = _signature(binding)
    with _LOCK:
        compiled = _CACHE.get(signature)
    if compiled is None:
        raise RuntimeError(
            "static FP8 quantizer is not prepared; prepare its execution before run"
        )
    _, args, _ = _launch(binding)
    run_compiled(compiled, args)
    return binding.output


def clear_caches() -> None:
    with _LOCK:
        _CACHE.clear()


__all__ = [
    "Binding",
    "StaticFp8QuantLauncher",
    "bind",
    "clear_caches",
    "compile",
    "resolve_static_fp8_quant_launcher",
    "run",
]
