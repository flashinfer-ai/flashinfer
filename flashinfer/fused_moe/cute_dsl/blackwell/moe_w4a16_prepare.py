# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Byte-preserving BF16 permutation for the split W4A16 regular-TMA path.

Only valid routed rows are copied, matching the native permutation contract.
Padding rows remain untouched and are masked by the existing GEMM epilogues.
Workspace remains owned by the caller; only compiled kernels are cached.
"""

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.cute_dsl.utils import current_cuda_stream, make_ptr


_kernel_cache: dict[tuple, Any] = {}


@cute.jit
def _vector(
    pointer: cute.Pointer,
    vector_words: cutlass.Constexpr,
    alignment: cutlass.Constexpr,
):
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint32,
            pointer.toint(),
            pointer.memspace,
            assumed_align=alignment,
        ),
        cute.make_layout((vector_words,)),
    )


class _MoeW4A16PermuteKernel:
    def __init__(self, row_bytes, route_tile, top_k, copy_bits, grid, enable_pdl):
        self.row_words = row_bytes // 4
        self.route_tile = route_tile
        self.top_k = top_k
        self.copy_bits = copy_bits
        self.vector_words = copy_bits // 32
        self.grid = grid
        self.enable_pdl = enable_pdl
        self.threads = 256

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        mapping_ptr: cute.Pointer,
        limits_ptr: cute.Pointer,
        num_tiles_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        self.kernel(
            input_ptr, output_ptr, mapping_ptr, limits_ptr, num_tiles_ptr
        ).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(self, input_ptr, output_ptr, mapping_ptr, limits_ptr, num_tiles_ptr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            # Routing is complete. Let FC1 load static weights while this
            # kernel copies B; FC1's activation wait protects every B load.
            cute.arch.griddepcontrol_launch_dependents()
        num_tiles = cute.make_tensor(num_tiles_ptr, cute.make_layout((1,)))[0]
        active_rows = num_tiles * self.route_tile
        mapping = cute.make_tensor(mapping_ptr, cute.make_layout((active_rows,)))
        limits = cute.make_tensor(limits_ptr, cute.make_layout((num_tiles,)))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint32,
            num_bits_per_copy=self.copy_bits,
        )
        values = cute.make_rmem_tensor((self.vector_words,), cutlass.Uint32)
        for row in cutlass.range(bidx, active_rows, self.grid):
            if row < limits[row // self.route_tile]:
                source = mapping[row] // self.top_k
                for iteration in cutlass.range_constexpr(
                    cute.ceil_div(self.row_words, self.threads * self.vector_words)
                ):
                    column = (tidx + iteration * self.threads) * self.vector_words
                    if column < self.row_words:
                        cute.copy(
                            copy_atom,
                            _vector(
                                input_ptr
                                + cutlass.Int64(source) * self.row_words
                                + column,
                                self.vector_words,
                                self.copy_bits // 8,
                            ),
                            values,
                        )
                        cute.copy(
                            copy_atom,
                            values,
                            _vector(
                                output_ptr
                                + cutlass.Int64(row) * self.row_words
                                + column,
                                self.vector_words,
                                self.copy_bits // 8,
                            ),
                        )


def moe_permute_w4a16(
    input: torch.Tensor,
    permuted_output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    max_num_permuted_tokens: int,
    top_k: int,
    tile_size: int,
    enable_pdl: bool,
) -> None:
    """Copy routed BF16 rows using compile-time top-k and aligned vectors."""
    if input.dtype != torch.bfloat16 or permuted_output.dtype != torch.bfloat16:
        raise TypeError("W4A16 permutation requires BF16 input and output")
    row_bytes = int(input.shape[1]) * input.element_size()
    if (
        row_bytes % 32 == 0
        and input.data_ptr() % 32 == 0
        and permuted_output.data_ptr() % 32 == 0
    ):
        copy_bits = 256
    elif (
        row_bytes % 16 == 0
        and input.data_ptr() % 16 == 0
        and permuted_output.data_ptr() % 16 == 0
    ):
        # Preserve the native permutation's 16-byte alignment contract for
        # contiguous input views that cannot use the wider copy.
        copy_bits = 128
    else:
        raise ValueError("W4A16 permutation requires 16-byte aligned rows and storage")
    if max_num_permuted_tokens == 0:
        return
    properties = torch.cuda.get_device_properties(input.device)
    resident_blocks = properties.max_threads_per_multi_processor // 256
    grid = min(
        max_num_permuted_tokens,
        properties.multi_processor_count * resident_blocks,
    )
    cache_key = (
        input.device.index,
        row_bytes,
        tile_size,
        top_k,
        copy_bits,
        grid,
        enable_pdl,
    )
    input_ptr = make_ptr(
        cutlass.Uint32,
        input.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=copy_bits // 8,
    )
    output_ptr = make_ptr(
        cutlass.Uint32,
        permuted_output.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=copy_bits // 8,
    )
    mapping_ptr = make_ptr(
        cutlass.Int32,
        permuted_idx_to_expanded_idx.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    limits_ptr = make_ptr(
        cutlass.Int32,
        tile_idx_to_mn_limit.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    num_tiles_ptr = make_ptr(
        cutlass.Int32,
        num_non_exiting_tiles.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    stream = current_cuda_stream()
    compiled = _kernel_cache.get(cache_key)
    if compiled is None:
        kernel = _MoeW4A16PermuteKernel(
            row_bytes, tile_size, top_k, copy_bits, grid, enable_pdl
        )
        compiled = cute.compile(
            kernel,
            input_ptr,
            output_ptr,
            mapping_ptr,
            limits_ptr,
            num_tiles_ptr,
            stream,
        )
        _kernel_cache[cache_key] = compiled
    compiled(input_ptr, output_ptr, mapping_ptr, limits_ptr, num_tiles_ptr, stream)
