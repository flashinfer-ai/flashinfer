"""Research-only CuTeDSL normalization stages for Qwen3.8 MTP feedback."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as compile_cute
from b12x._lib.compiler import run_compiled
from b12x._lib.intrinsics import block_reduce, warp_reduce
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

_THREADS = 256
_WARPS = _THREADS // 32
_LOCK = RLock()
_CACHE: dict[tuple[object, ...], Callable[..., None]] = {}
_WARMED: dict[tuple[object, ...], Callable[..., None]] = {}


def _add(left: Float32, right: Float32) -> Float32:
    return left + right


@cute.jit
def _sum_block(value: Float32, reduction: cute.Tensor) -> Float32:
    return block_reduce(
        warp_reduce(value, _add),
        _add,
        reduction,
        Float32(0.0),
    )


class _TokenNorm:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = int(hidden_size)
        self.items = (self.hidden_size + _THREADS - 1) // _THREADS

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        eps: Float32,
        tokens: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(source, weight, output, eps).launch(
            grid=(tokens, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        eps: Float32,
    ) -> None:
        token, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread_i = Int32(thread)
        base = Int64(token) * Int64(self.hidden_size)
        square_sum = Float32(0.0)
        for item in cutlass.range_constexpr(self.items):
            column = thread_i + Int32(item * _THREADS)
            if column < Int32(self.hidden_size):
                value = Float32(source[base + column.to(Int64)])
                square_sum += value * value

        allocator = cutlass.utils.SmemAllocator()
        reduction = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1, _WARPS)),
            byte_alignment=16,
        )
        inverse = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )
        total = _sum_block(square_sum, reduction)
        if thread_i == Int32(0):
            inverse[0] = cute.math.rsqrt(
                total / Float32(self.hidden_size) + eps,
                fastmath=True,
            )
        cute.arch.sync_threads()
        inv_rms = Float32(inverse[0])
        for item in cutlass.range_constexpr(self.items):
            column = thread_i + Int32(item * _THREADS)
            if column < Int32(self.hidden_size):
                offset = base + column.to(Int64)
                output[offset] = BFloat16(
                    Float32(source[offset])
                    * inv_rms
                    * (Float32(1.0) + Float32(weight[column.to(Int64)]))
                )


class _StateNorm:
    def __init__(self, streams: int, hidden_size: int) -> None:
        self.streams = int(streams)
        self.hidden_size = int(hidden_size)
        self.items = (self.hidden_size + _THREADS - 1) // _THREADS

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        eps: Float32,
        tokens: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(source, weight, output, eps).launch(
            grid=(tokens, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        eps: Float32,
    ) -> None:
        token, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread_i = Int32(thread)
        token_base = Int64(token) * Int64(self.streams * self.hidden_size)
        square_sum = Float32(0.0)
        for residual_stream in cutlass.range_constexpr(self.streams):
            stream_base = token_base + Int64(residual_stream * self.hidden_size)
            for item in cutlass.range_constexpr(self.items):
                column = thread_i + Int32(item * _THREADS)
                if column < Int32(self.hidden_size):
                    value = Float32(source[stream_base + column.to(Int64)])
                    square_sum += value * value

        allocator = cutlass.utils.SmemAllocator()
        reduction = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1, _WARPS)),
            byte_alignment=16,
        )
        inverse = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )
        total = _sum_block(square_sum, reduction)
        if thread_i == Int32(0):
            inverse[0] = cute.math.rsqrt(
                total / Float32(self.streams * self.hidden_size) + eps,
                fastmath=True,
            )
        cute.arch.sync_threads()
        inv_rms = Float32(inverse[0])
        for residual_stream in cutlass.range_constexpr(self.streams):
            stream_base = token_base + Int64(residual_stream * self.hidden_size)
            weight_base = Int64(residual_stream * self.hidden_size)
            for item in cutlass.range_constexpr(self.items):
                column = thread_i + Int32(item * _THREADS)
                if column < Int32(self.hidden_size):
                    offset = stream_base + column.to(Int64)
                    output[offset] = BFloat16(
                        Float32(source[offset])
                        * inv_rms
                        * (
                            Float32(1.0)
                            + Float32(weight[weight_base + column.to(Int64)])
                        )
                    )


def _pointer(tensor: torch.Tensor) -> cute.Pointer:
    return make_ptr(
        BFloat16,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=2,
    )


def _fake_pointer() -> cute.Pointer:
    return make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2)


def _launch(
    kind: str,
    entry: object,
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    *,
    tokens: int,
    streams: int,
    hidden_size: int,
) -> None:
    device_index = source.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (kind, int(device_index), int(streams), int(hidden_size))
    with torch.cuda.device(device_index):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            compiled = _CACHE.get(key)
            warmed = compiled is not None and _WARMED.get(key) is compiled
        if capturing and not warmed:
            raise RuntimeError(
                "MTP CuTe normalization kernels must be compiled and warm-run "
                "before CUDA graph capture"
            )
        if compiled is None:
            with _LOCK:
                compiled = _CACHE.get(key)
                if compiled is None:
                    raise_if_kernel_resolution_frozen(
                        "cute.compile", target=entry, cache_key=key
                    )
                    compiled = compile_cute(
                        entry,
                        _fake_pointer(),
                        _fake_pointer(),
                        _fake_pointer(),
                        Float32(1.0e-6),
                        Int32(1),
                        current_cuda_stream(),
                        compile_spec=KernelCompileSpec.from_key(
                            "sequence.mtp_feedback.norm",
                            2,
                            key,
                        ),
                    )
                    _CACHE[key] = compiled
        run_compiled(
            compiled,
            (
                _pointer(source),
                _pointer(weight),
                _pointer(output),
                float(eps),
                int(tokens),
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _CACHE.get(key) is compiled:
                _WARMED[key] = compiled


def token_norm(
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    eps: float,
    hidden_size: int,
) -> None:
    tokens = int(source.shape[0])
    _launch(
        "token",
        _TokenNorm(hidden_size),
        source,
        weight,
        output,
        eps,
        tokens=tokens,
        streams=1,
        hidden_size=hidden_size,
    )


def state_norm(
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    eps: float,
    streams: int,
    hidden_size: int,
) -> None:
    tokens = int(source.shape[0])
    _launch(
        "state",
        _StateNorm(streams, hidden_size),
        source,
        weight,
        output,
        eps,
        tokens=tokens,
        streams=streams,
        hidden_size=hidden_size,
    )


def clear_caches() -> None:
    with _LOCK:
        _CACHE.clear()
        _WARMED.clear()


__all__ = ["clear_caches", "state_norm", "token_norm"]
