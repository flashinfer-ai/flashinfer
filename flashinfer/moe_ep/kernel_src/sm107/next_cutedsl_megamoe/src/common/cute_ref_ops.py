# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Current-stream CuTe kernels shared by host quantizers and references."""

from __future__ import annotations

import functools
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr
from cutlass.cute.typing import AddressSpace, sym_int64
from cutlass.cutlass_dsl import Float32, Int32, Int64, T

_threads_per_block = 256


def _target_architecture() -> str:
    configured = os.environ.get("CUTE_DSL_ARCH")
    if configured:
        return configured
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _current_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _fake_vector(dtype: type, elements, alignment: int):
    return make_fake_compact_tensor(dtype, (elements,), stride_order=(0,), assumed_align=alignment)


@cute.kernel
def _rcp_approx_kernel(source: cute.Tensor, destination: cute.Tensor, element_count: Int64):
    thread_index, _, _ = cute.arch.thread_idx()
    block_index, _, _ = cute.arch.block_idx()
    element_index = Int64(block_index) * Int64(_threads_per_block) + Int64(thread_index)
    if element_index < element_count:
        destination[element_index] = cute.arch.rcp_approx(Float32(source[element_index]))


@cute.jit
def _rcp_approx_launcher(source: cute.Tensor, destination: cute.Tensor, stream: cuda.CUstream):
    element_count = cute.size(source)
    _rcp_approx_kernel(source, destination, Int64(element_count)).launch(
        grid=((element_count + _threads_per_block - 1) // _threads_per_block, 1, 1),
        block=(_threads_per_block, 1, 1),
        stream=stream,
    )


@functools.lru_cache(None)
def _compile_rcp_approx(_architecture: str, _device: int):
    elements = sym_int64(divisibility=1)
    source = _fake_vector(cutlass.Float32, elements, 4)
    destination = _fake_vector(cutlass.Float32, elements, 16)
    return cute.compile[cute.EnableTVMFFI(True)](
        _rcp_approx_launcher,
        source,
        destination,
        make_fake_stream(),
    )


def rcp_approx_ftz_f32(source: torch.Tensor) -> torch.Tensor:
    """Apply ``rcp.approx.ftz.f32`` to a contiguous CUDA FP32 tensor."""
    source_contiguous = source.contiguous()
    destination = torch.empty_like(source_contiguous)
    if source_contiguous.numel() == 0:
        return destination.view_as(source)
    with torch.cuda.device(source.device):
        compiled = _compile_rcp_approx(_target_architecture(), torch.cuda.current_device())
        compiled(source_contiguous.reshape(-1), destination.reshape(-1), _current_stream())
    return destination.view_as(source)


@cute.kernel
def _pack_fp4_kernel(source: cute.Pointer, destination: cute.Pointer, pair_count: Int64):
    thread_index, _, _ = cute.arch.thread_idx()
    block_index, _, _ = cute.arch.block_idx()
    pair_index = Int64(block_index) * Int64(_threads_per_block) + Int64(thread_index)
    if pair_index < pair_count:
        even = Float32((source + pair_index * Int64(2)).load())
        odd = Float32((source + pair_index * Int64(2) + Int64(1)).load())
        packed = Int32(
            llvm.inline_asm(
                T.i32(),
                [odd.ir_value(), even.ir_value()],
                "{\n"
                "  .reg .b8 converted;\n"
                "  cvt.rn.satfinite.e2m1x2.f32 converted, $1, $2;\n"
                "  mov.b32 $0, {converted, converted, converted, converted};\n"
                "}",
                "=r,f,f",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )
        (destination + pair_index).store(cutlass.Uint8(packed & Int32(0xFF)))


@cute.jit
def _pack_fp4_launcher(
    source: cute.Pointer,
    destination: cute.Pointer,
    pair_count: Int64,
    stream: cuda.CUstream,
):
    _pack_fp4_kernel(source, destination, pair_count).launch(
        grid=((pair_count + _threads_per_block - 1) // _threads_per_block, 1, 1),
        block=(_threads_per_block, 1, 1),
        stream=stream,
    )


@functools.lru_cache(None)
def _compile_pack_fp4(_architecture: str, _device: int):
    # Pointer arguments avoid CuTe layout's i32 shape limit for references larger than 2^31 elements.
    return cute.compile[cute.EnableTVMFFI(True)](
        _pack_fp4_launcher,
        make_ptr(cutlass.Float32, 0, AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=16),
        Int64(0),
        make_fake_stream(),
    )


def pack_f32_to_fp4_u8(source: torch.Tensor) -> torch.Tensor:
    """Pack contiguous CUDA FP32 pairs into raw E2M1 bytes."""
    source_contiguous = source.contiguous()
    if source_contiguous.numel() % 2 != 0:
        raise ValueError("FP4 packing requires an even number of elements.")
    destination = torch.empty(source_contiguous.numel() // 2, dtype=torch.uint8, device=source.device)
    if destination.numel() == 0:
        return destination
    with torch.cuda.device(source.device):
        compiled = _compile_pack_fp4(_target_architecture(), torch.cuda.current_device())
        compiled(
            source_contiguous.data_ptr(),
            destination.data_ptr(),
            destination.numel(),
            _current_stream(),
        )
    return destination


@cute.kernel
def _round_e8m0_kernel(source: cute.Tensor, destination: cute.Tensor, element_count: Int64):
    thread_index, _, _ = cute.arch.thread_idx()
    block_index, _, _ = cute.arch.block_idx()
    pair_index = Int64(block_index) * Int64(_threads_per_block) + Int64(thread_index)
    even_index = pair_index * Int64(2)
    if even_index < element_count:
        even = Float32(source[even_index])
        odd = Float32(0.0)
        if even_index + Int64(1) < element_count:
            odd = Float32(source[even_index + Int64(1)])
        packed = Int32(
            llvm.inline_asm(
                T.i32(),
                [odd.ir_value(), even.ir_value()],
                "{\n"
                "  .reg .b16 converted;\n"
                "  cvt.rp.satfinite.ue8m0x2.f32 converted, $1, $2;\n"
                "  mov.b32 $0, {converted, converted};\n"
                "}",
                "=r,f,f",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )
        destination[even_index] = cutlass.Uint8(packed & Int32(0xFF))
        if even_index + Int64(1) < element_count:
            destination[even_index + Int64(1)] = cutlass.Uint8((packed >> Int32(8)) & Int32(0xFF))


@cute.jit
def _round_e8m0_launcher(source: cute.Tensor, destination: cute.Tensor, stream: cuda.CUstream):
    element_count = cute.size(source)
    pair_count = (element_count + 1) // 2
    _round_e8m0_kernel(source, destination, Int64(element_count)).launch(
        grid=((pair_count + _threads_per_block - 1) // _threads_per_block, 1, 1),
        block=(_threads_per_block, 1, 1),
        stream=stream,
    )


@functools.lru_cache(None)
def _compile_round_e8m0(_architecture: str, _device: int):
    elements = sym_int64(divisibility=1)
    source = _fake_vector(cutlass.Float32, elements, 4)
    destination = _fake_vector(cutlass.Uint8, elements, 16)
    return cute.compile[cute.EnableTVMFFI(True)](
        _round_e8m0_launcher,
        source,
        destination,
        make_fake_stream(),
    )


def round_f32_to_e8m0_u8(source: torch.Tensor) -> torch.Tensor:
    """Round contiguous CUDA FP32 values to raw E8M0 bytes."""
    source_contiguous = source.contiguous()
    destination = torch.empty(source_contiguous.numel(), dtype=torch.uint8, device=source.device)
    if source_contiguous.numel() == 0:
        return destination
    with torch.cuda.device(source.device):
        compiled = _compile_round_e8m0(_target_architecture(), torch.cuda.current_device())
        compiled(source_contiguous.reshape(-1), destination, _current_stream())
    return destination
