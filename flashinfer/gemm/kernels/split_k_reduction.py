# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small FP32 split-K reduction used by the CuTe DSL dense GEMM path."""

import functools
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.cute.arch import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)


_THREADS = 256
_VALUES_PER_THREAD = 4


@cute.kernel
def _split_k_reduce_kernel(
    partials: cute.Tensor,
    out: cute.Tensor,
    numel: cutlass.Int64,
    split_k_slices: cutlass.Constexpr[int],
    enable_pdl: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(enable_pdl):
        griddepcontrol_wait()

    block_idx, _, _ = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    block_base = cutlass.Int64(block_idx) * (
        _THREADS * _VALUES_PER_THREAD
    ) + cutlass.Int64(thread_idx)

    # Keep each iteration coalesced across the CTA. The partial buffer is laid
    # out as [split_k, M, N], so the same flattened output index is separated by
    # exactly ``numel`` elements between K slices.
    for value_idx in cutlass.range_constexpr(_VALUES_PER_THREAD):
        output_idx = block_base + value_idx * _THREADS
        if output_idx < numel:
            value = cutlass.Float32(0.0)
            for split_idx in cutlass.range_constexpr(split_k_slices):
                partial_idx = cutlass.Int64(split_idx) * numel + output_idx
                value += partials[partial_idx]
            out[output_idx] = value.to(out.element_type)

    if cutlass.const_expr(enable_pdl):
        griddepcontrol_launch_dependents()


@cute.jit
def _run_split_k_reduce(
    partials: cute.Tensor,
    out: cute.Tensor,
    numel: cutlass.Int64,
    split_k_slices: cutlass.Constexpr[int],
    enable_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    _split_k_reduce_kernel(
        partials,
        out,
        numel,
        split_k_slices,
        enable_pdl,
    ).launch(
        grid=(
            cute.ceil_div(numel, _THREADS * _VALUES_PER_THREAD),
            1,
            1,
        ),
        block=(_THREADS, 1, 1),
        stream=stream,
        use_pdl=enable_pdl,
    )


@functools.cache
def get_compiled_split_k_reduction(
    out_dtype: Type[cutlass.Numeric],
    split_k_slices: int,
    enable_pdl: bool,
    sm_version: int,
):
    """Compile one output-dtype/split specialization.

    ``sm_version`` participates in the Python cache key because CuTe DSL obtains
    the actual architecture from the active compilation context.
    """

    if split_k_slices not in (2, 4):
        raise ValueError(f"split_k_slices must be 2 or 4, got {split_k_slices}")
    _ = sm_version

    partial_size = cute.sym_int(64)
    output_size = cute.sym_int(64)
    partials_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (partial_size,),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        out_dtype,
        (output_size,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        _run_split_k_reduce,
        partials_fake,
        out_fake,
        cutlass.Int64(1),
        split_k_slices,
        enable_pdl,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )
