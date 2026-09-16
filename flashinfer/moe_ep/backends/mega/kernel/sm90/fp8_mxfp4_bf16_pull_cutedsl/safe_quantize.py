# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fuse post-amax MXFP4 activation quantization without changing normal rounding.

This module does not compute the row reduction. The caller supplies contiguous
FP32 input and its existing Torch absmax. Compilation is cached by static hidden
width and device, with symbolic token count and the TVM-FFI current stream.
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


class _SafeE4M3Quantize:
    def __init__(self, hidden: int):
        self.hidden = hidden

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        amax: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        stream,
    ):
        elements = Int64(cute.size(x, mode=[0])) * Int64(self.hidden)
        self.kernel(x, amax, payload, scales).launch(
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

            # Torch CUDA division by a CPU scalar uses multiplication by its
            # FP32 reciprocal. div.rn(amax, 448) can differ by one ULP here.
            d = _mul_rn(row_amax, Float32(1.0 / _E4M3_MAX))
            if d < Float32(_LEGACY_EPS):
                d = Float32(_LEGACY_EPS)

            scaled = Float32(0.0)
            if (row_amax > Float32(0.0)) & (row_amax < Float32(_TINY_AMAX)):
                safe_amax = row_amax
                if safe_amax < Float32(_MIN_AMAX):
                    safe_amax = Float32(_MIN_AMAX)
                q = _div_rn(Float32(_E4M3_MAX), safe_amax)
                if q > Float32(_FP32_MAX):
                    q = Float32(_FP32_MAX)
                d = _div_rn(Float32(1.0), q)
                scaled = _mul_rn(value, q)
            else:
                # Do not replace this tensor/tensor division with value*rcp(d).
                scaled = _div_rn(value, d)

            payload[row, col] = scaled.to(cutlass.Float8E4M3FN)
            if col == Int64(0):
                scales[row, 0] = d


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


__all__ = ["quantize_from_amax"]
