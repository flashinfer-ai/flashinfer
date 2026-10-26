# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Byte-preserving routed activation and canonical scale-factor preparation.

Copy valid routed payload rows without quantization or arithmetic. Invalid
payload rows remain untouched; downstream epilogues mask them from outputs.
Scale factors for invalid rows and physical row padding
are zeroed through the final 128-row SF atom required by TMA. Storage beyond
that physical extent remains untouched. Per-token scales and routing metadata
retain their original values and indexing.

The host allocates workspace per invocation and launches preparation before
FC1 on the same stream, including during CUDA graph capture. Only compiled
kernels are cached; workspace tensors follow the caller's allocation lifetime.
"""

from typing import Any
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from flashinfer.cute_dsl.utils import current_cuda_stream, make_ptr
from .utils import release_gpu

_CACHE: dict[tuple, Any] = {}


@cute.jit
def _vector(pointer):
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint32, pointer.toint(), pointer.memspace, assumed_align=32
        ),
        cute.make_layout((8,)),
    )


class PrepareRouted:
    def __init__(self, payload_bytes, sf_columns, route_tile, top_k, grid, enable_pdl):
        if payload_bytes % 32 or sf_columns % 4:
            raise ValueError(
                "payload rows require 32-byte alignment; SF rows require 4-scale groups"
            )
        self.payload_words = payload_bytes // 4
        self.sf_blocks = sf_columns // 4
        self.route_tile = route_tile
        self.top_k = top_k
        self.grid = grid
        self.enable_pdl = enable_pdl
        self.threads = 128

    @cute.jit
    def __call__(
        self,
        payload_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        routed_ptr: cute.Pointer,
        routed_sf_ptr: cute.Pointer,
        mapping_ptr: cute.Pointer,
        limits_ptr: cute.Pointer,
        num_tiles_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        self.kernel(
            payload_ptr,
            sf_ptr,
            routed_ptr,
            routed_sf_ptr,
            mapping_ptr,
            limits_ptr,
            num_tiles_ptr,
        ).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        payload_ptr,
        sf_ptr,
        routed_ptr,
        routed_sf_ptr,
        mapping_ptr,
        limits_ptr,
        num_tiles_ptr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            # Publish completed routing ancestors for the next grid's weight
            # prefetch. Its wait still protects this grid's payload/SF writes.
            if tidx < 32:
                if bidx == 0:
                    release_gpu()
                cute.arch.griddepcontrol_launch_dependents()
        count = cute.make_tensor(num_tiles_ptr, cute.make_layout((1,)))[0]
        active_rows = count * self.route_tile
        sf_rows = cute.ceil_div(active_rows, 128) * 128
        mapping = cute.make_tensor(mapping_ptr, cute.make_layout((active_rows,)))
        limits = cute.make_tensor(limits_ptr, cute.make_layout((count,)))
        copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=256
        )
        values = cute.make_rmem_tensor((8,), cutlass.Uint32)
        for row in cutlass.range(bidx, sf_rows, self.grid):
            valid = cutlass.Boolean(False)
            source = cutlass.Int32(0)
            if row < active_rows:
                valid = row < limits[row // self.route_tile]
                if valid:
                    source = mapping[row] // self.top_k
            if valid:
                for iteration in cutlass.range_constexpr(
                    cute.ceil_div(self.payload_words, self.threads * 8)
                ):
                    column = (tidx + iteration * self.threads) * 8
                    if column < self.payload_words:
                        cute.copy(
                            copy,
                            _vector(
                                payload_ptr
                                + cutlass.Int64(source) * self.payload_words
                                + column
                            ),
                            values,
                        )
                        cute.copy(
                            copy,
                            values,
                            _vector(
                                routed_ptr
                                + cutlass.Int64(row) * self.payload_words
                                + column
                            ),
                        )
            for iteration in cutlass.range_constexpr(
                cute.ceil_div(self.sf_blocks, self.threads)
            ):
                block = tidx + iteration * self.threads
                if block < self.sf_blocks:
                    value = cutlass.Uint32(0)
                    if valid:
                        src = cute.make_tensor(
                            sf_ptr + cutlass.Int64(source) * self.sf_blocks + block,
                            cute.make_layout((1,)),
                        )
                        value = cutlass.Uint32(src[0])
                    offset = (
                        cutlass.Int64(row // 128) * self.sf_blocks * 128
                        + block * 128
                        + (row % 32) * 4
                        + (row % 128) // 32
                    )
                    target = cute.make_tensor(
                        routed_sf_ptr + offset, cute.make_layout((1,))
                    )
                    target[0] = value


def allocate(activation, activation_sf, capacity, sf_vec_size):
    if activation.dtype == torch.uint8:
        reduction = activation.shape[1] * 2
    elif activation.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        reduction = activation.shape[1]
    else:
        raise TypeError(f"unsupported blockscaled activation dtype {activation.dtype}")
    if sf_vec_size not in (16, 32) or reduction % (sf_vec_size * 4):
        raise ValueError("unsupported SF geometry")
    payload = torch.empty(
        (capacity, activation.shape[1]),
        device=activation.device,
        dtype=activation.dtype,
    )
    sf = torch.empty(
        ((capacity + 127) // 128 * 128 * (reduction // sf_vec_size),),
        device=activation.device,
        dtype=activation_sf.dtype,
    )
    return payload, sf


def prepare(
    activation,
    activation_sf,
    routed,
    routed_sf,
    mapping,
    limits,
    num_tiles,
    route_tile,
    top_k,
    sf_vec_size,
    enable_pdl=True,
):
    if activation.dtype == torch.uint8:
        reduction = activation.shape[1] * 2
    elif activation.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        reduction = activation.shape[1]
    else:
        raise TypeError(f"unsupported activation dtype {activation.dtype}")
    sf_columns = reduction // sf_vec_size
    payload_bytes = activation.shape[1] * activation.element_size()
    capacity = mapping.numel()
    tensors = (activation, activation_sf, routed, routed_sf, mapping, limits, num_tiles)
    if any(
        not x.is_cuda or x.device != activation.device or not x.is_contiguous()
        for x in tensors
    ):
        raise ValueError("same-device contiguous CUDA tensors required")
    if (
        routed.shape != (capacity, activation.shape[1])
        or routed.dtype != activation.dtype
    ):
        raise ValueError("routed payload shape/dtype mismatch")
    if (
        activation_sf.numel() != activation.shape[0] * sf_columns
        or activation_sf.element_size() != 1
    ):
        raise ValueError("source SF must be byte-valued linear per-token scales")
    if (
        routed_sf.numel() != ((capacity + 127) // 128) * 128 * sf_columns
        or routed_sf.dtype != activation_sf.dtype
    ):
        raise ValueError("canonical routed SF storage mismatch")
    if any(x.dtype != torch.int32 for x in (mapping, limits, num_tiles)):
        raise TypeError("routing metadata must be int32")
    if (
        capacity % route_tile
        or limits.numel() < capacity // route_tile
        or num_tiles.numel() != 1
    ):
        raise ValueError("metadata capacity mismatch")
    if any(x.data_ptr() % 32 for x in (activation, routed)) or any(
        x.data_ptr() % 4 for x in (activation_sf, routed_sf)
    ):
        raise ValueError("unaligned preparation storage")
    if sf_vec_size not in (16, 32) or reduction % (4 * sf_vec_size):
        raise ValueError("unsupported SF geometry")
    # Bound the persistent grid by hardware thread capacity, as in the
    # W4A16 permutation launch. Workspace allocation remains per invocation.
    properties = torch.cuda.get_device_properties(activation.device)
    resident_blocks = properties.max_threads_per_multi_processor // 128
    grid = min(
        (capacity + 127) // 128 * 128,
        properties.multi_processor_count * resident_blocks,
    )
    key = (
        activation.device.index,
        payload_bytes,
        sf_columns,
        route_tile,
        top_k,
        grid,
        enable_pdl,
    )
    ptrs = tuple(
        make_ptr(
            cutlass.Uint32 if index < 4 else cutlass.Int32,
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32 if index in (0, 2) else 4,
        )
        for index, x in enumerate(tensors)
    )
    stream = current_cuda_stream()
    compiled = _CACHE.get(key)
    if compiled is None:
        kernel = PrepareRouted(
            payload_bytes, sf_columns, route_tile, top_k, grid, enable_pdl
        )
        compiled = cute.compile(kernel, *ptrs, stream)
        _CACHE[key] = compiled
    compiled(*ptrs, stream)
    return routed, routed_sf
