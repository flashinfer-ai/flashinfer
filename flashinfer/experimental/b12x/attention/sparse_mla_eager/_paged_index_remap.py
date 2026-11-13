"""Native request-relative to physical-slot remapping for paged sparse MLA."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.compiler import key_field, run_compiled
from b12x._lib.utils import current_cuda_stream

_LOCK = RLock()
_CACHE: dict[tuple[int, int, bool], object] = {}
BLOCK_SIZE = 64
TOPK = 2048
THREADS = 32
INDICES_PER_THREAD = TOPK // THREADS


def _flat_i32(tensor: torch.Tensor):
    converted = from_dlpack(tensor.reshape(-1), assumed_align=4)
    converted.element_type = cutlass.Int32
    return converted.mark_layout_dynamic(leading_dim=0)


class _RemapKernel:
    def __init__(self, max_q_rows: int, request_relative: bool):
        self.max_q_rows = int(max_q_rows)
        self.request_relative = bool(request_relative)

    @cute.jit
    def __call__(
        self,
        request_ids: cute.Tensor,
        block_table: cute.Tensor,
        logical_indices: cute.Tensor,
        physical_indices: cute.Tensor,
        input_counts: cute.Tensor,
        selected_counts: cute.Tensor,
        active_rows: Int32,
        request_count: Int32,
        table_width: Int32,
        num_cache_blocks: Int32,
        block_stride_records: Int64,
        token_stride_records: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            request_ids,
            block_table,
            logical_indices,
            physical_indices,
            input_counts,
            selected_counts,
            active_rows,
            request_count,
            table_width,
            num_cache_blocks,
            block_stride_records,
            token_stride_records,
        ).launch(
            grid=(self.max_q_rows, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        request_ids: cute.Tensor,
        block_table: cute.Tensor,
        logical_indices: cute.Tensor,
        physical_indices: cute.Tensor,
        input_counts: cute.Tensor,
        selected_counts: cute.Tensor,
        active_rows: Int32,
        request_count: Int32,
        table_width: Int32,
        num_cache_blocks: Int32,
        block_stride_records: Int64,
        token_stride_records: Int64,
    ):
        row_idx, _, _ = cute.arch.block_idx()
        row = Int32(row_idx)
        if row < active_rows:
            thread = Int32(cute.arch.thread_idx()[0])
            request = Int32(0)
            if cutlass.const_expr(self.request_relative):
                request = request_ids[row].to(Int32)
            column_begin = thread * Int32(INDICES_PER_THREAD)
            count = Int32(0)
            for offset in cutlass.range(Int32(INDICES_PER_THREAD), unroll=1):
                column = column_begin + offset
                logical_offset = Int64(row) * Int64(TOPK) + Int64(column)
                logical_slot = logical_indices[logical_offset].to(Int32)
                logical_block = logical_slot // Int32(BLOCK_SIZE)
                valid = (logical_slot >= Int32(0)) & (logical_block >= Int32(0))
                if cutlass.const_expr(self.request_relative):
                    valid = (
                        valid
                        & (request >= Int32(0))
                        & (request < request_count)
                        & (logical_block < table_width)
                    )
                else:
                    valid = valid & (column < input_counts[row].to(Int32))
                if valid:
                    physical_block = logical_block
                    if cutlass.const_expr(self.request_relative):
                        table_offset = request.to(Int64) * table_width.to(
                            Int64
                        ) + logical_block.to(Int64)
                        physical_block = block_table[table_offset].to(Int32)
                    if (physical_block >= Int32(0)) & (
                        physical_block < num_cache_blocks
                    ):
                        count += Int32(1)

            # The first warp of the output row is temporary storage for the
            # per-thread counts. Every thread loads its stable prefix before
            # any thread overwrites the temporary values.
            row_offset = Int64(row) * Int64(TOPK)
            physical_indices[row_offset + thread.to(Int64)] = count
            cute.arch.sync_threads()
            output_base = Int32(0)
            total = Int32(0)
            for peer in cutlass.range_constexpr(THREADS):
                peer_count = physical_indices[row_offset + Int64(peer)].to(Int32)
                if Int32(peer) < thread:
                    output_base += peer_count
                total += peer_count
            cute.arch.sync_threads()
            if thread == Int32(0):
                selected_counts[row] = total

            local_output = Int32(0)
            for offset in cutlass.range(Int32(INDICES_PER_THREAD), unroll=1):
                column = column_begin + offset
                logical_offset = row_offset + column.to(Int64)
                logical_slot = logical_indices[logical_offset].to(Int32)
                logical_block = logical_slot // Int32(BLOCK_SIZE)
                valid = (logical_slot >= Int32(0)) & (logical_block >= Int32(0))
                if cutlass.const_expr(self.request_relative):
                    valid = (
                        valid
                        & (request >= Int32(0))
                        & (request < request_count)
                        & (logical_block < table_width)
                    )
                else:
                    valid = valid & (column < input_counts[row].to(Int32))
                if valid:
                    physical_block = logical_block
                    if cutlass.const_expr(self.request_relative):
                        table_offset = request.to(Int64) * table_width.to(
                            Int64
                        ) + logical_block.to(Int64)
                        physical_block = block_table[table_offset].to(Int32)
                    if (physical_block >= Int32(0)) & (
                        physical_block < num_cache_blocks
                    ):
                        physical_record = (
                            physical_block.to(Int64) * block_stride_records
                            + (logical_slot % Int32(BLOCK_SIZE)).to(Int64)
                            * token_stride_records
                        )
                        output_offset = (
                            row_offset + output_base.to(Int64) + local_output.to(Int64)
                        )
                        physical_indices[output_offset] = physical_record.to(Int32)
                        local_output += Int32(1)


@dataclass(frozen=True)
class Binding:
    request_ids: torch.Tensor
    block_table: torch.Tensor
    logical_indices: torch.Tensor
    physical_indices: torch.Tensor
    input_counts: torch.Tensor
    selected_counts: torch.Tensor
    max_q_rows: int
    num_cache_blocks: int
    block_stride_records: int
    token_stride_records: int
    request_relative: bool


def bind(
    *,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    logical_indices: torch.Tensor,
    physical_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    max_q_rows: int,
    num_cache_blocks: int,
    block_stride_records: int,
    token_stride_records: int,
) -> Binding:
    rows = int(logical_indices.shape[0])
    tensors = (
        request_ids,
        block_table,
        logical_indices,
        physical_indices,
        selected_counts,
    )
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError("sparse index metadata must use int32 tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("sparse index metadata must be contiguous")
    if any(tensor.device != logical_indices.device for tensor in tensors):
        raise ValueError("sparse index metadata must share one device")
    if tuple(logical_indices.shape) != (rows, TOPK):
        raise ValueError(f"logical_indices must have shape ({rows}, {TOPK})")
    if tuple(physical_indices.shape) != (rows, TOPK):
        raise ValueError(f"physical_indices must have shape ({rows}, {TOPK})")
    if tuple(request_ids.shape) != (rows,):
        raise ValueError(f"request_ids must have shape ({rows},)")
    if tuple(selected_counts.shape) != (rows,):
        raise ValueError(f"selected_counts must have shape ({rows},)")
    if block_table.ndim != 2 or block_table.shape[0] <= 0 or block_table.shape[1] <= 0:
        raise ValueError("block_table must be a non-empty two-dimensional tensor")
    if not 0 < rows <= int(max_q_rows):
        raise ValueError("logical index rows exceed the planned capacity")
    if not 0 < int(num_cache_blocks) <= torch.iinfo(torch.int32).max // BLOCK_SIZE:
        raise ValueError("num_cache_blocks exceeds the physical-slot int32 range")
    if int(block_stride_records) <= 0 or int(token_stride_records) <= 0:
        raise ValueError("physical record strides must be positive")
    return Binding(
        request_ids=request_ids.detach(),
        block_table=block_table.detach(),
        logical_indices=logical_indices.detach(),
        physical_indices=physical_indices.detach(),
        input_counts=selected_counts.detach(),
        selected_counts=selected_counts.detach(),
        max_q_rows=int(max_q_rows),
        num_cache_blocks=int(num_cache_blocks),
        block_stride_records=int(block_stride_records),
        token_stride_records=int(token_stride_records),
        request_relative=True,
    )


def bind_physical_slots(
    *,
    physical_slots: torch.Tensor,
    input_counts: torch.Tensor,
    physical_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    max_q_rows: int,
    num_cache_blocks: int,
    block_stride_records: int,
    token_stride_records: int,
) -> Binding:
    rows = int(physical_slots.shape[0])
    tensors = (physical_slots, input_counts, physical_indices, selected_counts)
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError("sparse index metadata must use int32 tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("sparse index metadata must be contiguous")
    if any(tensor.device != physical_slots.device for tensor in tensors):
        raise ValueError("sparse index metadata must share one device")
    if tuple(physical_slots.shape) != (rows, TOPK):
        raise ValueError(f"physical_slots must have shape ({rows}, {TOPK})")
    if tuple(physical_indices.shape) != (rows, TOPK):
        raise ValueError(f"physical_indices must have shape ({rows}, {TOPK})")
    if tuple(input_counts.shape) != (rows,) or tuple(selected_counts.shape) != (rows,):
        raise ValueError(f"sparse counts must have shape ({rows},)")
    if not 0 < rows <= int(max_q_rows):
        raise ValueError("physical slot rows exceed the planned capacity")
    if not 0 < int(num_cache_blocks) <= torch.iinfo(torch.int32).max // BLOCK_SIZE:
        raise ValueError("num_cache_blocks exceeds the physical-slot int32 range")
    if int(block_stride_records) <= 0 or int(token_stride_records) <= 0:
        raise ValueError("physical record strides must be positive")
    placeholder = input_counts
    return Binding(
        request_ids=placeholder.detach(),
        block_table=placeholder.detach(),
        logical_indices=physical_slots.detach(),
        physical_indices=physical_indices.detach(),
        input_counts=input_counts.detach(),
        selected_counts=selected_counts.detach(),
        max_q_rows=int(max_q_rows),
        num_cache_blocks=int(num_cache_blocks),
        block_stride_records=int(block_stride_records),
        token_stride_records=int(token_stride_records),
        request_relative=False,
    )


def _signature(binding: Binding) -> tuple[int, int, bool]:
    index = binding.logical_indices.device.index
    if index is None:
        index = torch.cuda.current_device()
    return int(index), int(binding.max_q_rows), bool(binding.request_relative)


def _launch(binding: Binding):
    entry = _RemapKernel(binding.max_q_rows, binding.request_relative)
    request_count = int(binding.block_table.shape[0]) if binding.request_relative else 1
    table_width = int(binding.block_table.shape[1]) if binding.request_relative else 1
    args = (
        _flat_i32(binding.request_ids),
        _flat_i32(binding.block_table),
        _flat_i32(binding.logical_indices),
        _flat_i32(binding.physical_indices),
        _flat_i32(binding.input_counts),
        _flat_i32(binding.selected_counts),
        Int32(binding.logical_indices.shape[0]),
        Int32(request_count),
        Int32(table_width),
        Int32(binding.num_cache_blocks),
        Int64(binding.block_stride_records),
        Int64(binding.token_stride_records),
        current_cuda_stream(),
    )
    spec = KernelCompileSpec.from_fields(
        "attention.paged_sparse_index_remap",
        2,
        key_field("max_q_rows", binding.max_q_rows),
        key_field("request_relative", binding.request_relative),
    )
    return entry, args, spec


def compile(*, binding: Binding) -> None:
    signature = _signature(binding)
    with _LOCK:
        compiled = _CACHE.get(signature)
    if compiled is None:
        entry, args, spec = _launch(binding)
        compiled = compile_cute(entry, *args, compile_spec=spec)
        with _LOCK:
            _CACHE[signature] = compiled


def run(*, binding: Binding) -> tuple[torch.Tensor, torch.Tensor]:
    signature = _signature(binding)
    with _LOCK:
        compiled = _CACHE.get(signature)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "sparse index remap compile miss during CUDA graph capture; "
                "call compile first"
            )
        compile(binding=binding)
        with _LOCK:
            compiled = _CACHE[signature]
    _, args, _ = _launch(binding)
    run_compiled(compiled, args)
    return binding.physical_indices, binding.selected_counts


def clear_caches() -> None:
    with _LOCK:
        _CACHE.clear()


__all__ = [
    "Binding",
    "bind",
    "bind_physical_slots",
    "clear_caches",
    "compile",
    "run",
]
