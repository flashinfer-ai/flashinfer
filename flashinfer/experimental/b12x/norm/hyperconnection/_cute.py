"""CuTeDSL multi-stream normalization and residual-injection kernels."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as compile_cute
from b12x._lib.compiler import run_compiled
from b12x._lib.intrinsics import (
    bfloat2_to_float2_scaled,
    block_reduce,
    fabs_f32,
    fmax_f32,
    ld_global_v4_u32,
    pack_f32x2_to_bfloat2,
    st_global_v4_u32,
    warp_reduce,
    u32_as_f32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x._lib.program_cache import register_program_cache

_THREADS = 256
_WARPS = _THREADS // 32
_LOCK = RLock()
_CACHE: dict[tuple[object, ...], Callable[..., None]] = {}
_WARMED: dict[tuple[object, ...], Callable[..., None]] = {}
register_program_cache(_CACHE, mirrors=(_WARMED,), lock=_LOCK)
_POINTER_DTYPES = {
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.bool: Boolean,
}


def _add(left: Float32, right: Float32) -> Float32:
    return left + right


@cute.jit
def _sigmoid(value: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + cute.math.exp(-value, fastmath=True))


@cute.jit
def _block_sum(value: Float32, reduction: cute.Tensor) -> Float32:
    return block_reduce(
        warp_reduce(value, _add),
        _add,
        reduction,
        Float32(0.0),
    )


@dsl_user_op
def _float_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.bitcast(
            T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


class _GroupedRmsNorm:
    def __init__(
        self, streams: int, hidden_size: int, zero_centered: bool = True
    ) -> None:
        self.streams = int(streams)
        self.hidden_size = int(hidden_size)
        self.items = (self.hidden_size + _THREADS - 1) // _THREADS
        self.zero_centered = bool(zero_centered)

    @cute.jit
    def __call__(
        self,
        state: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        eps: Float32,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(state, weight, output, eps).launch(
            grid=(rows, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        state: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        eps: Float32,
    ) -> None:
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        row_i = Int64(row)
        thread_i = Int32(thread)
        base = row_i * Int64(self.hidden_size)
        weight_base = Int64(Int32(row) % Int32(self.streams)) * Int64(self.hidden_size)

        values = cute.make_rmem_tensor((self.items,), Float32)
        square_sum = Float32(0.0)
        for item in cutlass.range_constexpr(self.items):
            column = thread_i + Int32(item * _THREADS)
            value = Float32(0.0)
            if column < Int32(self.hidden_size):
                value = Float32(state[base + column.to(Int64)])
            values[item] = value
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
        total = _block_sum(square_sum, reduction)
        if thread_i == Int32(0):
            inverse[0] = cute.math.rsqrt(
                total / Float32(self.hidden_size) + eps,
                fastmath=True,
            )
        cute.arch.sync_threads()
        scale = Float32(inverse[0])
        for item in cutlass.range_constexpr(self.items):
            column = thread_i + Int32(item * _THREADS)
            if column < Int32(self.hidden_size):
                learned = Float32(weight[weight_base + column.to(Int64)])
                output[base + column.to(Int64)] = BFloat16(
                    values[item]
                    * scale
                    * (Float32(1.0 if self.zero_centered else 0.0) + learned)
                )


class _EngramMix:
    """Per-stream normalized-dot gate over the existing multi-stream residual."""

    def __init__(self, streams: int, hidden_size: int, has_mask: bool) -> None:
        self.streams = int(streams)
        self.hidden_size = int(hidden_size)
        self.has_mask = bool(has_mask)
        self.items = (self.hidden_size + _THREADS - 1) // _THREADS

    @cute.jit
    def __call__(
        self,
        state: cute.Pointer,
        projected_kv: cute.Pointer,
        norm_weights: cute.Pointer,
        token_mask: cute.Pointer,
        output: cute.Pointer,
        eps: Float32,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            state, projected_kv, norm_weights, token_mask, output, eps
        ).launch(
            grid=(rows * Int32(self.streams), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        state: cute.Pointer,
        projected_kv: cute.Pointer,
        norm_weights: cute.Pointer,
        token_mask: cute.Pointer,
        output: cute.Pointer,
        eps: Float32,
    ) -> None:
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        token = Int64(row) // Int64(self.streams)
        stream_id = Int64(row) % Int64(self.streams)
        base = Int64(row) * Int64(self.hidden_size)
        key_base = (
            token * Int64((self.streams + 1) * self.hidden_size)
            + stream_id * Int64(self.hidden_size)
        )
        value_base = (
            token * Int64((self.streams + 1) * self.hidden_size)
            + Int64(self.streams * self.hidden_size)
        )
        weight_base = stream_id * Int64(self.hidden_size)
        values = cute.make_rmem_tensor((self.items,), Float32)
        h_square = Float32(0.0)
        k_square = Float32(0.0)
        product = Float32(0.0)
        for item in cutlass.range_constexpr(self.items):
            column = Int32(thread) + Int32(item * _THREADS)
            h = Float32(0.0)
            k = Float32(0.0)
            w = Float32(0.0)
            if column < Int32(self.hidden_size):
                offset = column.to(Int64)
                h = Float32(state[base + offset])
                k = Float32(projected_kv[key_base + offset])
                w = Float32(norm_weights[weight_base + offset])
            values[item] = h
            h_square += h * h
            k_square += k * k
            product += (h * w) * k

        allocator = cutlass.utils.SmemAllocator()
        h_reduction = allocator.allocate_tensor(
            Float32, cute.make_layout((1, _WARPS)), byte_alignment=16
        )
        k_reduction = allocator.allocate_tensor(
            Float32, cute.make_layout((1, _WARPS)), byte_alignment=16
        )
        dot_reduction = allocator.allocate_tensor(
            Float32, cute.make_layout((1, _WARPS)), byte_alignment=16
        )
        gate_storage = allocator.allocate_tensor(
            Float32, cute.make_layout((1,)), byte_alignment=4
        )
        h_total = _block_sum(h_square, h_reduction)
        k_total = _block_sum(k_square, k_reduction)
        dot_total = _block_sum(product, dot_reduction)
        if Int32(thread) == Int32(0):
            rstd = cute.math.rsqrt(
                h_total / Float32(self.hidden_size) + eps, fastmath=True
            ) * cute.math.rsqrt(
                k_total / Float32(self.hidden_size) + eps, fastmath=True
            )
            dot = dot_total * rstd * Float32(self.hidden_size ** -0.5)
            magnitude = cute.math.sqrt(
                fmax_f32(fabs_f32(dot), Float32(1.0e-6)), fastmath=True
            )
            # copysign, not sign(dot): +0 must yield sigmoid(+sqrt(1e-6)).
            signed = u32_as_f32(
                (_float_bits(magnitude) & Uint32(0x7FFFFFFF))
                | (_float_bits(dot) & Uint32(0x80000000))
            )
            gate = _sigmoid(signed)
            if cutlass.const_expr(self.has_mask):
                if not token_mask[token]:
                    gate = Float32(0.0)
            gate_storage[0] = gate
        cute.arch.sync_threads()
        gate = Float32(gate_storage[0])
        for item in cutlass.range_constexpr(self.items):
            column = Int32(thread) + Int32(item * _THREADS)
            if column < Int32(self.hidden_size):
                offset = column.to(Int64)
                value = Float32(projected_kv[value_base + offset])
                output[base + offset] = BFloat16(values[item] + gate * value)


class _ScaledSilu:
    def __init__(self, streams: int, lowrank: int) -> None:
        self.streams = int(streams)
        self.lowrank = int(lowrank)

    @cute.jit
    def __call__(
        self,
        projected: cute.Pointer,
        output: cute.Pointer,
        elements: Int32,
        projected_stride: Int64,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(projected, output, elements, projected_stride).launch(
            grid=((elements + Int32(_THREADS - 1)) // Int32(_THREADS), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        projected: cute.Pointer,
        output: cute.Pointer,
        elements: Int32,
        projected_stride: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        offset = Int64(block) * Int64(_THREADS) + Int64(thread)
        if offset < elements.to(Int64):
            source = (offset // Int64(self.lowrank)) * projected_stride + (
                offset % Int64(self.lowrank)
            )
            scaled_bf16 = BFloat16(Float32(projected[source]) / Float32(self.streams))
            scaled = Float32(scaled_bf16)
            output[offset] = BFloat16(scaled * _sigmoid(scaled))


class _Pointwise:
    """Unscaled activation/residual tails with one final destination cast."""

    def __init__(
        self, operation: str, output_dtype: type, width: int = 1,
        limit: float = float("inf"), round_silu: bool = False,
    ) -> None:
        self.operation = operation
        self.output_dtype = output_dtype
        self.width = int(width)
        self.limit = float(limit)
        self.round_silu = bool(round_silu)

    @cute.jit
    def __call__(
        self, left: cute.Pointer, right: cute.Pointer, output: cute.Pointer,
        elements: Int64, stream: cuda.CUstream,
    ) -> None:
        self.kernel(left, right, output, elements).launch(
            grid=((elements + Int64(_THREADS - 1)) // Int64(_THREADS), 1, 1),
            block=(_THREADS, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(
        self, left: cute.Pointer, right: cute.Pointer, output: cute.Pointer,
        elements: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        offset = Int64(block) * Int64(_THREADS) + Int64(thread)
        if offset < elements:
            if cutlass.const_expr(self.operation == "swiglu"):
                source = (offset // Int64(self.width)) * Int64(2 * self.width)
                source += offset % Int64(self.width)
                gate = Float32(left[source])
                up = Float32(left[source + Int64(self.width)])
                if gate > Float32(self.limit):
                    gate = Float32(self.limit)
                if up > Float32(self.limit):
                    up = Float32(self.limit)
                if up < Float32(-self.limit):
                    up = Float32(-self.limit)
                activated = gate * _sigmoid(gate)
                if cutlass.const_expr(self.round_silu):
                    activated = Float32(BFloat16(activated))
                value = activated * up
            elif cutlass.const_expr(self.operation == "add"):
                value = Float32(left[offset]) + Float32(right[offset])
            else:
                value = _sigmoid(Float32(left[offset]))
            output[offset] = value.to(self.output_dtype)


class _GateMean:
    def __init__(self, streams: int, hidden_size: int) -> None:
        self.streams = int(streams)
        self.hidden_size = int(hidden_size)
        self.column_blocks = (self.hidden_size + _THREADS - 1) // _THREADS

    @cute.jit
    def __call__(
        self,
        normalized: cute.Pointer,
        logits: cute.Pointer,
        output: cute.Pointer,
        tokens: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(normalized, logits, output).launch(
            grid=(tokens, self.column_blocks, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        normalized: cute.Pointer,
        logits: cute.Pointer,
        output: cute.Pointer,
    ) -> None:
        token, column_block, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        column = Int32(column_block) * Int32(_THREADS) + Int32(thread)
        if column < Int32(self.hidden_size):
            base = Int64(token) * Int64(self.streams * self.hidden_size)
            total = Float32(0.0)
            for residual_stream in cutlass.range_constexpr(self.streams):
                offset = (
                    base + Int64(residual_stream * self.hidden_size) + column.to(Int64)
                )
                value = BFloat16(normalized[offset])
                gate = BFloat16(_sigmoid(Float32(logits[offset])))
                total += Float32(BFloat16(Float32(value) * Float32(gate)))
            output[Int64(token) * Int64(self.hidden_size) + column.to(Int64)] = (
                BFloat16(total / Float32(self.streams))
            )


class _Combine:
    def __init__(self, streams: int, hidden_size: int) -> None:
        self.streams = int(streams)
        self.hidden_size = int(hidden_size)
        self.column_blocks = (self.hidden_size + _THREADS - 1) // _THREADS

    @cute.jit
    def __call__(
        self,
        state: cute.Pointer,
        block_output: cute.Pointer,
        logits: cute.Pointer,
        combined: cute.Pointer,
        tokens: Int32,
        logit_stride: Int64,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(state, block_output, logits, combined, logit_stride).launch(
            grid=(tokens, self.streams, self.column_blocks),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        state: cute.Pointer,
        block_output: cute.Pointer,
        logits: cute.Pointer,
        combined: cute.Pointer,
        logit_stride: Int64,
    ) -> None:
        token, residual_stream, column_block = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        column = Int32(column_block) * Int32(_THREADS) + Int32(thread)
        if column < Int32(self.hidden_size):
            state_offset = (
                Int64(token) * Int64(self.streams) + Int64(residual_stream)
            ) * Int64(self.hidden_size) + column.to(Int64)
            output_offset = Int64(token) * Int64(self.hidden_size) + column.to(Int64)
            logit_offset = Int64(token) * logit_stride + Int64(residual_stream)
            scaled_logit = Float32(logits[logit_offset]) / Float32(self.streams)
            scale = Float32(2.0) * cute.math.rcp(
                Float32(1.0) + cute.math.exp(-scaled_logit, fastmath=True),
                approx=True,
                ftz=True,
            )
            combined[state_offset] = BFloat16(
                cute.math.fma(
                    scale,
                    Float32(block_output[output_offset]),
                    Float32(state[state_offset]),
                )
            )


class _CombineNorm:
    def __init__(self, streams: int, hidden_size: int) -> None:
        self.streams = int(streams)
        self.hidden_size = int(hidden_size)
        self.items = (self.hidden_size + _THREADS - 1) // _THREADS

    @cute.jit
    def __call__(
        self,
        state: cute.Pointer,
        block_output: cute.Pointer,
        logits: cute.Pointer,
        weight: cute.Pointer,
        combined: cute.Pointer,
        normalized: cute.Pointer,
        eps: Float32,
        tokens: Int32,
        logit_stride: Int64,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            state,
            block_output,
            logits,
            weight,
            combined,
            normalized,
            eps,
            logit_stride,
        ).launch(
            grid=(tokens, self.streams, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        state: cute.Pointer,
        block_output: cute.Pointer,
        logits: cute.Pointer,
        weight: cute.Pointer,
        combined: cute.Pointer,
        normalized: cute.Pointer,
        eps: Float32,
        logit_stride: Int64,
    ) -> None:
        token, residual_stream, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread_i = Int32(thread)
        state_base = (
            Int64(token) * Int64(self.streams) + Int64(residual_stream)
        ) * Int64(self.hidden_size)
        output_base = Int64(token) * Int64(self.hidden_size)
        weight_base = Int64(residual_stream) * Int64(self.hidden_size)
        logit_offset = Int64(token) * logit_stride + Int64(residual_stream)
        scale = Float32(2.0) * _sigmoid(
            Float32(logits[logit_offset]) / Float32(self.streams)
        )

        rounded = cute.make_rmem_tensor((self.items,), Float32)
        square_sum = Float32(0.0)
        for item in cutlass.range_constexpr(self.items):
            column = thread_i + Int32(item * _THREADS)
            value = BFloat16(0.0)
            if column < Int32(self.hidden_size):
                offset = state_base + column.to(Int64)
                value = BFloat16(
                    Float32(state[offset])
                    + scale * Float32(block_output[output_base + column.to(Int64)])
                )
                combined[offset] = value
            rounded[item] = Float32(value)
            square_sum += Float32(value) * Float32(value)

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
        total = _block_sum(square_sum, reduction)
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
                learned = Float32(weight[weight_base + column.to(Int64)])
                normalized[state_base + column.to(Int64)] = BFloat16(
                    rounded[item] * inv_rms * (Float32(1.0) + learned)
                )


class _PackedCombineNorm:
    _PACK_ELEMENTS = 8
    _THREADS = 256
    _WARPS = _THREADS // 32

    def __init__(self, streams: int, hidden_size: int) -> None:
        self.streams = int(streams)
        self.hidden_size = int(hidden_size)
        self.packs = self.hidden_size // self._PACK_ELEMENTS
        self.items = (self.packs + self._THREADS - 1) // self._THREADS

    @cute.jit
    def __call__(
        self,
        state: cute.Pointer,
        block_output: cute.Pointer,
        logits: cute.Pointer,
        weight: cute.Pointer,
        combined: cute.Pointer,
        normalized: cute.Pointer,
        eps: Float32,
        tokens: Int32,
        logit_stride: Int64,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            state,
            block_output,
            logits,
            weight,
            combined,
            normalized,
            eps,
            logit_stride,
        ).launch(
            grid=(tokens, self.streams, 1),
            block=(self._THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        state: cute.Pointer,
        block_output: cute.Pointer,
        logits: cute.Pointer,
        weight: cute.Pointer,
        combined: cute.Pointer,
        normalized: cute.Pointer,
        eps: Float32,
        logit_stride: Int64,
    ) -> None:
        token, residual_stream, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread_i = Int32(thread)
        allocator = cutlass.utils.SmemAllocator()
        reduction = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1, self._WARPS)),
            byte_alignment=16,
        )
        inverse = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )

        lane = thread_i % Int32(32)
        scale = Float32(0.0)
        if lane == Int32(0):
            logit_offset = Int64(token) * logit_stride + Int64(residual_stream)
            scale = Float32(2.0) * _sigmoid(
                Float32(logits[logit_offset]) / Float32(self.streams)
            )
        scale = Float32(cute.arch.shuffle_sync(scale, Int32(0)))

        state_base = Int64(state.toint()) + (
            (Int64(token) * Int64(self.streams) + Int64(residual_stream))
            * Int64(self.hidden_size * 2)
        )
        output_base = Int64(block_output.toint()) + Int64(token) * Int64(
            self.hidden_size * 2
        )
        weight_base = Int64(weight.toint()) + Int64(residual_stream) * Int64(
            self.hidden_size * 2
        )
        combined_base = Int64(combined.toint()) + (
            (Int64(token) * Int64(self.streams) + Int64(residual_stream))
            * Int64(self.hidden_size * 2)
        )
        normalized_base = Int64(normalized.toint()) + (
            (Int64(token) * Int64(self.streams) + Int64(residual_stream))
            * Int64(self.hidden_size * 2)
        )

        rounded = cute.make_rmem_tensor((self.items, self._PACK_ELEMENTS), Float32)
        square_sum = Float32(0.0)
        for item in cutlass.range_constexpr(self.items):
            for element in cutlass.range_constexpr(self._PACK_ELEMENTS):
                rounded[item, element] = Float32(0.0)
            pack = thread_i + Int32(item * self._THREADS)
            if pack < Int32(self.packs):
                byte_offset = pack.to(Int64) * Int64(16)
                state_words = ld_global_v4_u32(state_base + byte_offset)
                output_words = ld_global_v4_u32(output_base + byte_offset)
                combined_words = cute.make_rmem_tensor((4,), Uint32)
                for word in cutlass.range_constexpr(4):
                    state_lo, state_hi = bfloat2_to_float2_scaled(
                        state_words[word], Float32(1.0)
                    )
                    output_lo, output_hi = bfloat2_to_float2_scaled(
                        output_words[word], Float32(1.0)
                    )
                    combined_words[word] = pack_f32x2_to_bfloat2(
                        state_lo + scale * output_lo,
                        state_hi + scale * output_hi,
                    )
                    value_lo, value_hi = bfloat2_to_float2_scaled(
                        combined_words[word], Float32(1.0)
                    )
                    element = word * 2
                    rounded[item, element] = value_lo
                    rounded[item, element + 1] = value_hi
                st_global_v4_u32(
                    combined_base + byte_offset,
                    combined_words[0],
                    combined_words[1],
                    combined_words[2],
                    combined_words[3],
                )
            for element in cutlass.range_constexpr(self._PACK_ELEMENTS):
                value = rounded[item, element]
                square_sum += value * value

        total = _block_sum(square_sum, reduction)
        if thread_i == Int32(0):
            inverse[0] = cute.math.rsqrt(
                total / Float32(self.hidden_size) + eps,
                fastmath=True,
            )
        cute.arch.sync_threads()
        inv_rms = Float32(inverse[0])

        for item in cutlass.range_constexpr(self.items):
            pack = thread_i + Int32(item * self._THREADS)
            if pack < Int32(self.packs):
                byte_offset = pack.to(Int64) * Int64(16)
                weight_words = ld_global_v4_u32(weight_base + byte_offset)
                normalized_words = cute.make_rmem_tensor((4,), Uint32)
                for word in cutlass.range_constexpr(4):
                    weight_lo, weight_hi = bfloat2_to_float2_scaled(
                        weight_words[word], Float32(1.0)
                    )
                    element = word * 2
                    normalized_words[word] = pack_f32x2_to_bfloat2(
                        rounded[item, element] * inv_rms * (Float32(1.0) + weight_lo),
                        rounded[item, element + 1]
                        * inv_rms
                        * (Float32(1.0) + weight_hi),
                    )
                st_global_v4_u32(
                    normalized_base + byte_offset,
                    normalized_words[0],
                    normalized_words[1],
                    normalized_words[2],
                    normalized_words[3],
                )


def _pointer(tensor: torch.Tensor) -> cute.Pointer:
    return make_ptr(
        _POINTER_DTYPES[tensor.dtype],
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=tensor.element_size(),
    )


def _fake_pointer(dtype: torch.dtype = torch.bfloat16) -> cute.Pointer:
    return make_ptr(
        _POINTER_DTYPES[dtype],
        16,
        cute.AddressSpace.gmem,
        assumed_align=1 if dtype == torch.bool else 4 if dtype == torch.float32 else 2,
    )


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else int(index)


def _compile(
    key: tuple[object, ...],
    entry: object,
    argument_count: int,
    device: torch.device,
    *,
    has_eps: bool = False,
    runtime_ints: int = 0,
    runtime_int64s: int = 0,
    pointer_dtypes: tuple[torch.dtype, ...] = (),
) -> Callable[..., None]:
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        with torch.cuda.device(device):
            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=entry,
                cache_key=key,
            )
            args: list[object] = [
                _fake_pointer(dtype)
                for dtype in (
                    pointer_dtypes or (torch.bfloat16,) * argument_count
                )
            ]
            if has_eps:
                args.append(Float32(1.0e-6))
            args.extend(Int32(1) for _ in range(runtime_ints))
            args.extend(Int64(1) for _ in range(runtime_int64s))
            args.append(current_cuda_stream())
            compiled = compile_cute(
                entry,
                *args,
                compile_spec=KernelCompileSpec.from_key(
                    "norm.hyperconnection.cute",
                    7,
                    key,
                ),
            )
        _CACHE[key] = compiled
        return compiled


def _run(
    key: tuple[object, ...],
    entry: object,
    tensors: tuple[torch.Tensor, ...],
    *,
    eps: float | None = None,
    runtime_ints: tuple[int, ...] = (),
    runtime_int64s: tuple[int, ...] = (),
) -> None:
    device = tensors[0].device
    with torch.cuda.device(device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            compiled = _CACHE.get(key)
            warmed = compiled is not None and _WARMED.get(key) is compiled
        if capturing and not warmed:
            raise RuntimeError(
                "HyperConnection CuTe kernels must be compiled and warm-run "
                "before CUDA graph capture"
            )
        if compiled is None:
            compiled = _compile(
                key,
                entry,
                len(tensors),
                device,
                has_eps=eps is not None,
                runtime_ints=len(runtime_ints),
                runtime_int64s=len(runtime_int64s),
                pointer_dtypes=tuple(tensor.dtype for tensor in tensors),
            )
        args: list[object] = [_pointer(tensor) for tensor in tensors]
        if eps is not None:
            args.append(float(eps))
        args.extend(int(value) for value in runtime_ints)
        args.extend(int(value) for value in runtime_int64s)
        args.append(current_cuda_stream())
        run_compiled(compiled, tuple(args))
    if not capturing:
        with _LOCK:
            if _CACHE.get(key) is compiled:
                _WARMED[key] = compiled


def grouped_rmsnorm(
    state: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    eps: float,
    streams: int,
    hidden_size: int,
    zero_centered: bool = True,
) -> None:
    rows = int(state.shape[0]) * int(streams)
    key = (
        "norm" if zero_centered else "rmsnorm",
        _device_index(state),
        int(streams),
        int(hidden_size),
    )
    if not zero_centered:
        key += (str(weight.dtype),)
    _run(
        key,
        _GroupedRmsNorm(streams, hidden_size, zero_centered),
        (state, weight, output),
        eps=eps,
        runtime_ints=(rows,),
    )


def scaled_silu(
    projected: torch.Tensor,
    output: torch.Tensor,
    *,
    streams: int,
) -> None:
    elements = int(projected.numel())
    lowrank = int(projected.shape[1])
    key = ("silu", _device_index(projected), int(streams), lowrank)
    _run(
        key,
        _ScaledSilu(streams, lowrank),
        (projected, output),
        runtime_ints=(elements,),
        runtime_int64s=(int(projected.stride(0)),),
    )


def gate_mean(
    normalized: torch.Tensor,
    logits: torch.Tensor,
    output: torch.Tensor,
    *,
    streams: int,
    hidden_size: int,
) -> None:
    tokens = int(normalized.shape[0])
    key = ("gate", _device_index(normalized), int(streams), int(hidden_size))
    _run(
        key,
        _GateMean(streams, hidden_size),
        (normalized, logits, output),
        runtime_ints=(tokens,),
    )


def combine(
    state: torch.Tensor,
    block_output: torch.Tensor,
    logits: torch.Tensor,
    combined: torch.Tensor,
    *,
    streams: int,
    hidden_size: int,
) -> None:
    tokens = int(state.shape[0])
    key = ("combine", _device_index(state), int(streams), int(hidden_size))
    _run(
        key,
        _Combine(streams, hidden_size),
        (state, block_output, logits, combined),
        runtime_ints=(tokens,),
        runtime_int64s=(int(logits.stride(0)),),
    )


def combine_norm(
    state: torch.Tensor,
    block_output: torch.Tensor,
    logits: torch.Tensor,
    weight: torch.Tensor,
    combined: torch.Tensor,
    normalized: torch.Tensor,
    *,
    eps: float,
    streams: int,
    hidden_size: int,
) -> None:
    tokens = int(state.shape[0])
    packed = int(hidden_size) % _PackedCombineNorm._PACK_ELEMENTS == 0
    key = (
        "combine_norm_packed" if packed else "combine_norm",
        _device_index(state),
        int(streams),
        int(hidden_size),
    )
    _run(
        key,
        (
            _PackedCombineNorm(streams, hidden_size)
            if packed
            else _CombineNorm(streams, hidden_size)
        ),
        (state, block_output, logits, weight, combined, normalized),
        eps=eps,
        runtime_ints=(tokens,),
        runtime_int64s=(int(logits.stride(0)),),
    )


def engram_mix(
    state: torch.Tensor,
    projected_kv: torch.Tensor,
    norm_weights: torch.Tensor,
    token_mask: torch.Tensor | None,
    output: torch.Tensor,
    *,
    eps: float,
    streams: int,
    hidden_size: int,
) -> None:
    key = (
        "engram_mix",
        _device_index(state),
        int(streams),
        int(hidden_size),
        token_mask is not None,
    )
    _run(
        key,
        _EngramMix(streams, hidden_size, token_mask is not None),
        (
            state,
            projected_kv,
            norm_weights,
            state if token_mask is None else token_mask,
            output,
        ),
        eps=eps,
        runtime_ints=(int(state.shape[0]),),
    )


def pointwise(
    operation: str,
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    *,
    width: int = 1,
    limit: float = float("inf"),
    round_silu: bool = False,
) -> None:
    key = (
        operation, _device_index(left), str(left.dtype), str(right.dtype), str(output.dtype),
        int(width), None if limit == float("inf") else float(limit), bool(round_silu),
    )
    _run(
        key,
        _Pointwise(operation, _POINTER_DTYPES[output.dtype], width, limit, round_silu),
        (left, right, output),
        runtime_int64s=(int(output.numel()),),
    )


def clear_caches() -> None:
    with _LOCK:
        _CACHE.clear()
        _WARMED.clear()


__all__ = [
    "clear_caches",
    "combine",
    "combine_norm",
    "engram_mix",
    "gate_mean",
    "grouped_rmsnorm",
    "scaled_silu",
]
