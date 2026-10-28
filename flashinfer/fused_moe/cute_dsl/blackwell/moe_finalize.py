# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic finalize for BF16 expert outputs in expanded or permuted order."""

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import torch

from flashinfer.cute_dsl.utils import (
    current_cuda_stream,
    make_ptr,
    torch_to_cutlass_dtype,
)


_kernel_cache: dict[tuple, Any] = {}


@dsl_user_op
def _mul_bf16_pair(word, scale, loc=None, ip=None):
    # A separate first multiply preserves signed zero, as in MegaMoE combine.
    return cutlass.Uint64(
        llvm.inline_asm(
            T.i64(),
            [
                cutlass.Uint32(word).ir_value(loc=loc, ip=ip),
                cutlass.Float32(scale).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b32 lo, hi; .reg .b64 pair, scales; "
            "shl.b32 lo, $1, 16; and.b32 hi, $1, 0xffff0000; "
            "mov.b64 pair, {lo, hi}; mov.b64 scales, {$2, $2}; "
            "mul.rn.f32x2 $0, pair, scales; }",
            "=l,r,f",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma_bf16_pair(word, scale, accum, loc=None, ip=None):
    return cutlass.Uint64(
        llvm.inline_asm(
            T.i64(),
            [
                cutlass.Uint32(word).ir_value(loc=loc, ip=ip),
                cutlass.Float32(scale).ir_value(loc=loc, ip=ip),
                cutlass.Uint64(accum).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b32 lo, hi; .reg .b64 pair, scales; "
            "shl.b32 lo, $1, 16; and.b32 hi, $1, 0xffff0000; "
            "mov.b64 pair, {lo, hi}; mov.b64 scales, {$2, $2}; "
            "fma.rn.f32x2 $0, pair, scales, $3; }",
            "=l,r,f,l",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _to_bf16_pair(accum, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint64(accum).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 lo, hi; mov.b64 {lo, hi}, $1; cvt.rn.bf16x2.f32 $0, hi, lo; }",
            "=r,l",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _bf16_vector(pointer):
    # Every row and per-thread offset is a multiple of eight BF16 elements.
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint32,
            pointer.toint(),
            pointer.memspace,
            assumed_align=16,
        ),
        cute.make_layout((4,)),
    )


class _MoeFinalizeKernel:
    def __init__(
        self,
        hidden_size,
        top_k,
        threads,
        enable_pdl,
        input_is_expanded,
        vectors_per_thread=1,
    ):
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.threads = threads
        self.vectors_per_thread = vectors_per_thread
        tile_width = threads * 8 * vectors_per_thread
        self.hidden_tiles = (hidden_size + tile_width - 1) // tile_width
        self.enable_pdl = enable_pdl
        self.input_is_expanded = input_is_expanded

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        mapping_ptr: cute.Pointer,
        scales_ptr: cute.Pointer,
        dense_indices_ptr,
        num_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        metadata_layout = cute.make_layout(
            (num_tokens, self.top_k), stride=(self.top_k, 1)
        )
        mapping = cute.make_tensor(mapping_ptr, metadata_layout)
        scales = cute.make_tensor(scales_ptr, metadata_layout)
        dense_indices = None
        if cutlass.const_expr(dense_indices_ptr is not None):
            dense_indices = cute.make_tensor(dense_indices_ptr, metadata_layout)
        self.kernel(input_ptr, output_ptr, mapping, scales, dense_indices).launch(
            grid=[num_tokens * self.hidden_tiles, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(self, input_ptr, output_ptr, mapping, scales, dense_indices):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        token_idx = bidx // self.hidden_tiles
        column = (
            bidx % self.hidden_tiles * self.threads * self.vectors_per_thread + tidx
        ) * 8
        route_base = cutlass.Int64(token_idx) * self.top_k

        rows = cute.make_rmem_tensor((self.top_k,), cutlass.Int32)
        weights = cute.make_rmem_tensor((self.top_k,), cutlass.Float32)
        accum = cute.make_rmem_tensor((4, self.vectors_per_thread), cutlass.Uint64)
        accum.fill(0)
        values = cute.make_rmem_tensor((4,), cutlass.Uint32)
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=128
        )

        # The caller guarantees routing is ready before FC2 signals us.
        # Only expert outputs depend on the still-running FC2 grid.
        for slot in cutlass.range_constexpr(self.top_k):
            if cutlass.const_expr(self.input_is_expanded):
                rows[slot] = cutlass.Int32(mapping[(token_idx, slot)] >= 0)
            else:
                rows[slot] = mapping[(token_idx, slot)]
            weights[slot] = cutlass.Float32(scales[(token_idx, slot)])

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        # Match MegaMoE's first-slot multiply and subsequent ordered FP32 FMA.
        # Mask before loading: nonlocal routes have no initialized expert row.
        for slot in cutlass.range_constexpr(self.top_k):
            row = cutlass.Int64(rows[slot])
            valid = rows[slot] >= 0
            if cutlass.const_expr(self.input_is_expanded):
                row = route_base + slot
                valid = rows[slot] != 0
            if valid:
                if cutlass.const_expr(dense_indices is not None):
                    # FC2 publishes these indices, so read only after its wait.
                    row = cutlass.Int64(dense_indices[token_idx, slot])
                for vector in cutlass.range_constexpr(self.vectors_per_thread):
                    vector_column = column + vector * self.threads * 8
                    if vector_column < self.hidden_size:
                        offset = row * self.hidden_size + vector_column
                        cute.copy(copy_atom, _bf16_vector(input_ptr + offset), values)
                        for element in cutlass.range_constexpr(4):
                            if cutlass.const_expr(slot == 0):
                                accum[element, vector] = _mul_bf16_pair(
                                    values[element], weights[slot]
                                )
                            else:
                                accum[element, vector] = _fma_bf16_pair(
                                    values[element],
                                    weights[slot],
                                    accum[element, vector],
                                )
        for vector in cutlass.range_constexpr(self.vectors_per_thread):
            vector_column = column + vector * self.threads * 8
            if vector_column < self.hidden_size:
                for element in cutlass.range_constexpr(4):
                    values[element] = _to_bf16_pair(accum[element, vector])
                offset = cutlass.Int64(token_idx) * self.hidden_size + vector_column
                cute.copy(copy_atom, values, _bf16_vector(output_ptr + offset))

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def moe_unpermute(
    permuted_input: torch.Tensor,
    output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    topk_scales: torch.Tensor,
    num_tokens: int,
    top_k: int,
    enable_pdl: bool = False,
    input_is_expanded: bool = False,
    dense_indices=None,
) -> None:
    """Reduce BF16 expert rows in ascending top-k order using FP32 FMA.

    Routing metadata must be ready before FC2 signals this dependent grid.
    Blockscaled FC2 may read routing before its dependency wait, using the
    prior grid's publication; it republishes routing before signaling combine.
    W4A16's earlier permute waits
    for sort before signaling FC1. Metadata reads precede this kernel's
    PDL wait; all expert-output reads follow it.
    Negative mapping entries skip their row entirely. An all-masked token
    writes exactly positive zero. Like MegaMoE's deterministic combine, slot
    zero uses an FP32 multiply and later slots use FP32 FMA, with round-to-nearest
    and subnormal preservation. This rounding contract applies to both W4A4
    and W4A16 callers.

    With ``input_is_expanded=True``, input rows are ``token * top_k + slot``;
    otherwise each nonnegative mapping value is the permuted input row.
    Input/output must be separate contiguous BF16 tensors, with a hidden
    dimension divisible by eight and 16-byte-aligned storage. Routing scales
    may be FP32, BF16, or FP16. Permuted row indices must be in bounds, as
    guaranteed by ``moe_sort``; no device values are read on the host.
    Token count is a runtime parameter of the cached kernel.
    """
    if num_tokens < 0 or top_k <= 0:
        raise ValueError("num_tokens must be nonnegative and top_k positive")
    if permuted_input.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise TypeError("MoE finalize requires BF16 input and output")
    if expanded_idx_to_permuted_idx.dtype != torch.int32:
        raise TypeError("route mapping must have dtype int32")
    if topk_scales.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("routing scales must be FP32, BF16, or FP16")
    tensors = (permuted_input, output, expanded_idx_to_permuted_idx, topk_scales)
    if any(t.ndim != 2 or not t.is_contiguous() for t in tensors):
        raise ValueError("MoE finalize requires contiguous matrices")
    if not output.is_cuda or any(t.device != output.device for t in tensors):
        raise ValueError("all MoE finalize tensors must share one CUDA device")
    hidden_size = output.shape[1]
    if hidden_size <= 0 or hidden_size % 8:
        raise ValueError("hidden_size must be positive and divisible by eight")
    if permuted_input.shape[1] != hidden_size or (
        input_is_expanded and permuted_input.shape[0] < num_tokens * top_k
    ):
        raise ValueError("input hidden size or expanded row count does not match")
    if output.shape[0] < num_tokens or any(
        t.shape[0] < num_tokens or t.shape[1] != top_k
        for t in (expanded_idx_to_permuted_idx, topk_scales)
    ):
        raise ValueError("output and routing metadata must cover num_tokens")
    if any(t.data_ptr() % 16 for t in (permuted_input, output)):
        raise ValueError("input and output must be 16-byte aligned")
    if num_tokens == 0:
        return

    # Spread short batches across SMs; amortize CTA and metadata overhead
    # over more columns once there are enough token rows to fill the GPU.
    threads = 32 if num_tokens <= 8 else 128
    vectors_per_thread = 4 if num_tokens > 1024 else 1
    if num_tokens > 1024 and input_is_expanded:
        if hidden_size % (threads * 8 * vectors_per_thread):
            # Avoid a partially occupied final CTA while bounding the
            # larger tile to seven vectors per thread.
            for candidate_vectors in (7, 6, 5):
                if hidden_size % (threads * 8 * candidate_vectors) == 0:
                    vectors_per_thread = candidate_vectors
                    break
    cache_key = (
        output.device.index,
        hidden_size,
        top_k,
        topk_scales.dtype,
        threads,
        vectors_per_thread,
        enable_pdl,
        input_is_expanded,
        dense_indices is not None,
    )
    with torch.cuda.device(output.device):
        stream = current_cuda_stream()
        pointers = (
            make_ptr(
                cutlass.BFloat16,
                permuted_input.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.BFloat16,
                output.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Int32,
                expanded_idx_to_permuted_idx.data_ptr(),
                cute.AddressSpace.gmem,
            ),
            make_ptr(
                torch_to_cutlass_dtype(topk_scales.dtype),
                topk_scales.data_ptr(),
                cute.AddressSpace.gmem,
            ),
            make_ptr(cutlass.Int32, dense_indices.data_ptr(), cute.AddressSpace.gmem)
            if dense_indices is not None
            else None,
        )
        compiled = _kernel_cache.get(cache_key)
        if compiled is None:
            kernel = _MoeFinalizeKernel(
                hidden_size,
                top_k,
                threads,
                enable_pdl,
                input_is_expanded,
                vectors_per_thread,
            )
            compiled = cute.compile(kernel, *pointers, num_tokens, stream)
            _kernel_cache[cache_key] = compiled
        compiled(*pointers, num_tokens, stream)
