# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Safe MXFP4 activation quantization and BF16 input staging.

The FP32 quantizer consumes a supplied absmax. The BF16 stager reduces each
finite input row directly before applying the same safe quantization math.
Compilation uses symbolic token counts and the TVM-FFI current stream.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from cutlass._mlir.dialects import llvm


_THREADS = 256
_E4M3_MAX = 448.0
_LEGACY_EPS = 1.0e-30
_TINY_AMAX = _E4M3_MAX * _LEGACY_EPS
_FP32_MAX = 3.4028234663852886e38
_MIN_AMAX = 1.3165537626040637e-36


@cute.jit
def _mul_rn(a: Float32, b: Float32):
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [a.ir_value(), b.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _div_rn(a: Float32, b: Float32):
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [a.ir_value(), b.ir_value()],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _safe_scale(row_amax: Float32):
    # Match the existing Torch scalar division and tiny-value rescaling exactly.
    d = _mul_rn(row_amax, Float32(1.0 / _E4M3_MAX))
    if d < Float32(_LEGACY_EPS):
        d = Float32(_LEGACY_EPS)
    tiny = (row_amax > Float32(0.0)) & (row_amax < Float32(_TINY_AMAX))
    q = Float32(0.0)
    if tiny:
        safe_amax = row_amax
        if safe_amax < Float32(_MIN_AMAX):
            safe_amax = Float32(_MIN_AMAX)
        q = _div_rn(Float32(_E4M3_MAX), safe_amax)
        if q > Float32(_FP32_MAX):
            q = Float32(_FP32_MAX)
        d = _div_rn(Float32(1.0), q)
    return d, q, tiny


@cute.jit
def _safe_scaled_value(value: Float32, d: Float32, q: Float32, tiny):
    scaled = Float32(0.0)
    if tiny:
        scaled = _mul_rn(value, q)
    else:
        scaled = _div_rn(value, d)
    return scaled


class _SafeE4M3Quantize:
    def __init__(self, hidden: int, top_k: int = 0):
        self.hidden = hidden
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        amax: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        stream,
        topk_ids=None,
        topk_weights=None,
        ids_out=None,
        weights_out=None,
    ):
        elements = Int64(cute.size(x, mode=[0])) * Int64(self.hidden)
        if cutlass.const_expr(self.top_k > 0):
            routes = Int64(cute.size(ids_out, mode=[0])) * Int64(self.top_k)
            if routes > elements:
                elements = routes
        self.kernel(
            x, amax, payload, scales, topk_ids, topk_weights, ids_out, weights_out
        ).launch(
            grid=((elements + _THREADS - 1) // _THREADS, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        amax: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        topk_ids=None,
        topk_weights=None,
        ids_out=None,
        weights_out=None,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = Int64(block) * Int64(_THREADS) + Int64(tid)
        elements = Int64(cute.size(x, mode=[0])) * Int64(self.hidden)
        if index < elements:
            row = index // Int64(self.hidden)
            col = index % Int64(self.hidden)
            row_amax = amax[row, 0]
            value = x[row, col]

            d, q, tiny = _safe_scale(row_amax)
            scaled = _safe_scaled_value(value, d, q, tiny)

            payload[row, col] = scaled.to(cutlass.Float8E4M3FN)
            if col == Int64(0):
                if cutlass.const_expr(self.top_k > 0):
                    for wire_col in cutlass.range_constexpr(4):
                        scales[row, wire_col] = d
                else:
                    scales[row, 0] = d

        if cutlass.const_expr(self.top_k > 0):
            routes = Int64(cute.size(ids_out, mode=[0])) * Int64(self.top_k)
            if index < routes:
                row = index // Int64(self.top_k)
                col = index % Int64(self.top_k)
                if row < Int64(cute.size(x, mode=[0])):
                    ids_out[row, col] = topk_ids[row, col]
                    weights_out[row, col] = topk_weights[row, col]
                else:
                    # Clear on every replay: the same workspace can be reused
                    # by graphs with different live token counts.
                    ids_out[row, col] = Int64(-1)


@functools.cache
def _compiled_quantizer(hidden: int, device_index: int):
    # Called under the input device guard; include the device in the cache so
    # a compiled module is not accidentally reused across device contexts.
    del device_index
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "safe E4M3 quantization requires one eager warmup before CUDA "
            "Graph capture; a cache miss cannot compile inside capture"
        )
    tokens = cute.sym_int()
    x = cute.runtime.make_fake_compact_tensor(
        Float32, (tokens, hidden), stride_order=(1, 0), assumed_align=4
    )
    amax = cute.runtime.make_fake_compact_tensor(
        Float32, (tokens, 1), stride_order=(1, 0), assumed_align=4
    )
    payload = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (tokens, hidden),
        stride_order=(1, 0),
        assumed_align=1,
    )
    scales = cute.runtime.make_fake_compact_tensor(
        Float32, (tokens, 1), stride_order=(1, 0), assumed_align=4
    )
    return cute.compile(
        _SafeE4M3Quantize(hidden),
        x,
        amax,
        payload,
        scales,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def quantize_from_amax(
    x: torch.Tensor, amax: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous CUDA FP32 [T,H] using supplied FP32 absmax [T,1].

    The caller guarantees finite inputs and a nonnegative, matching row absmax.
    No host-side value inspection or CUDA synchronization is performed. Empty
    batches return empty payload/scales without compilation or a kernel launch.
    """
    if x.ndim != 2 or x.shape[1] <= 0:
        raise ValueError("x must have shape [tokens, positive hidden]")
    if amax.shape != (x.shape[0], 1):
        raise ValueError("amax must have shape [tokens, 1]")
    if x.dtype != torch.float32 or amax.dtype != torch.float32:
        raise ValueError("x and amax must both be FP32")
    if not x.is_contiguous() or not amax.is_contiguous():
        raise ValueError("x and amax must both be contiguous")
    if not x.is_cuda or x.device != amax.device:
        raise ValueError("x and amax must be on the same CUDA device")

    payload = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(amax.shape, dtype=torch.float32, device=x.device)
    if x.shape[0] == 0:
        return payload, scales
    with torch.cuda.device(x.device):
        compiled = _compiled_quantizer(x.shape[1], x.device.index)
        compiled(x, amax, payload, scales)
    return payload, scales


@functools.cache
def _compiled_stager(hidden: int, top_k: int, device_index: int):
    del device_index
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "safe E4M3 staging requires one eager warmup before CUDA Graph capture"
        )
    tokens, capacity = cute.sym_int(), cute.sym_int()

    def tensor(dtype, shape, alignment):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=(1, 0), assumed_align=alignment
        )

    return cute.compile(
        _SafeE4M3Quantize(hidden, top_k),
        tensor(Float32, (tokens, hidden), 4),
        tensor(Float32, (tokens, 1), 4),
        tensor(cutlass.Float8E4M3FN, (capacity, hidden), 1),
        tensor(Float32, (capacity, 4), 4),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        tensor(Int64, (tokens, top_k), 8),
        tensor(Float32, (tokens, top_k), 4),
        tensor(Int64, (capacity, top_k), 8),
        tensor(Float32, (capacity, top_k), 4),
        options="--enable-tvm-ffi",
    )


def quantize_and_stage_from_amax(
    x, amax, payload, scales, topk_ids, topk_weights, ids_out, weights_out
):
    """Write quantized data and routing directly to validated contiguous buffers.

    The staging caller validates the shapes, dtypes, device and capacity before
    selecting this path. Arithmetic is shared with ``quantize_from_amax``.
    """
    with torch.cuda.device(x.device):
        compiled = _compiled_stager(x.shape[1], topk_ids.shape[1], x.device.index)
        compiled(x, amax, payload, scales, topk_ids, topk_weights, ids_out, weights_out)


class _BF16SafeStager:
    """One CTA per live row, with separate CTAs clearing padded routing rows."""

    def __init__(self, hidden: int, top_k: int):
        self.hidden = hidden
        self.top_k = top_k
        self.threads = 256
        self.slots = (hidden + self.threads - 1) // self.threads

    @cute.jit
    def __call__(
        self, x, payload, scales, topk_ids, topk_weights, ids_out, weights_out, stream
    ):
        tokens = Int64(cute.size(x, mode=[0]))
        tail = (Int64(cute.size(ids_out, mode=[0])) - tokens) * self.top_k
        self.kernel(
            x, payload, scales, topk_ids, topk_weights, ids_out, weights_out
        ).launch(
            grid=(tokens + (tail + self.threads - 1) // self.threads, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x, payload, scales, topk_ids, topk_weights, ids_out, weights_out):
        tid = cute.arch.thread_idx()[0]
        row = cute.arch.block_idx()[0]
        tokens = cute.size(x, mode=[0])
        if row < tokens:
            values = cute.make_rmem_tensor((self.slots,), Float32)
            local_max = Float32(0.0)
            for slot in cutlass.range_constexpr(self.slots):
                col = tid + slot * self.threads
                value = Float32(0.0)
                if col < self.hidden:
                    value = Float32(x[row, col])
                values[slot] = value
                magnitude = Float32(
                    llvm.inline_asm(
                        Float32.mlir_type,
                        [value.ir_value()],
                        "abs.f32 $0, $1;",
                        "=f,f",
                        has_side_effects=False,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )
                if magnitude > local_max:
                    local_max = magnitude
            for index in cutlass.range_constexpr(5):
                peer = cute.arch.shuffle_sync_bfly(local_max, offset=1 << index)
                if peer > local_max:
                    local_max = peer
            smem = cutlass.utils.SmemAllocator()
            partials = smem.allocate_array(Float32, self.threads // 32)
            lane = tid % 32
            if lane == 0:
                partials[tid // 32] = local_max
            cute.arch.sync_threads()
            row_amax = Float32(0.0)
            if lane < self.threads // 32:
                row_amax = partials[lane]
            for index in cutlass.range_constexpr(5):
                peer = cute.arch.shuffle_sync_bfly(row_amax, offset=1 << index)
                if peer > row_amax:
                    row_amax = peer
            d, q, tiny = _safe_scale(row_amax)
            for slot in cutlass.range_constexpr(self.slots):
                col = tid + slot * self.threads
                if col < self.hidden:
                    scaled = _safe_scaled_value(values[slot], d, q, tiny)
                    payload[row, col] = scaled.to(cutlass.Float8E4M3FN)
            if tid < 4:
                scales[row, tid] = d
            for slot in cutlass.range_constexpr(
                (self.top_k + self.threads - 1) // self.threads
            ):
                col = tid + slot * self.threads
                if col < self.top_k:
                    ids_out[row, col] = topk_ids[row, col]
                    weights_out[row, col] = topk_weights[row, col]
        else:
            index = Int64(row - tokens) * self.threads + tid
            tail = Int64(cute.size(ids_out, mode=[0]) - tokens) * self.top_k
            if index < tail:
                ids_out[tokens + index // self.top_k, index % self.top_k] = Int64(-1)


@functools.cache
def _compiled_bf16_stager(hidden: int, top_k: int, device_index: int):
    del device_index
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "BF16 safe staging requires eager warmup before CUDA Graph capture"
        )
    tokens, capacity = cute.sym_int(), cute.sym_int()

    def tensor(dtype, shape, alignment):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=(1, 0), assumed_align=alignment
        )

    return cute.compile(
        _BF16SafeStager(hidden, top_k),
        tensor(cutlass.BFloat16, (tokens, hidden), 2),
        tensor(cutlass.Float8E4M3FN, (capacity, hidden), 1),
        tensor(Float32, (capacity, 4), 4),
        tensor(Int64, (tokens, top_k), 8),
        tensor(Float32, (tokens, top_k), 4),
        tensor(Int64, (capacity, top_k), 8),
        tensor(Float32, (capacity, top_k), 4),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def quantize_and_stage_bf16(
    x, payload, scales, topk_ids, topk_weights, ids_out, weights_out
):
    """Read finite BF16 rows, reduce amax and stage into validated buffers."""
    with torch.cuda.device(x.device):
        compiled = _compiled_bf16_stager(x.shape[1], topk_ids.shape[1], x.device.index)
        compiled(x, payload, scales, topk_ids, topk_weights, ids_out, weights_out)


__all__ = ["quantize_from_amax"]
