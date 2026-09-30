"""Allocation-free logical-to-physical V4.1 indexed-cache mapping."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import KernelCompileSpec, launch
from b12x._lib.utils import current_cuda_stream


@cute.kernel
def _map_indexed_pages_kernel(
    indices: cute.Tensor,
    lengths: cute.Tensor,
    page_table: cute.Tensor,
    output: cute.Tensor,
    rows: Int32,
    width: Int32,
    table_width: Int32,
    table_stride: Int64,
    page_size: Int64,
    num_pages: Int64,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    linear = Int64(block) * Int64(256) + Int64(thread)
    if linear < Int64(rows) * Int64(width):
        row = linear // Int64(width)
        column = linear - row * Int64(width)
        physical = Int32(-1)
        logical = Int64(indices[linear])
        if (column < Int64(lengths[row])) & (logical >= Int64(0)):
            page = logical // page_size
            if page < Int64(table_width):
                pid = Int64(page_table[row * table_stride + page])
                if (pid >= Int64(0)) & (pid < num_pages):
                    slot = pid * page_size + logical % page_size
                    if slot <= Int64(2147483647):
                        physical = Int32(slot)
        output[linear] = physical


@cute.jit
def _map_indexed_pages_launch(
    indices: cute.Tensor,
    lengths: cute.Tensor,
    page_table: cute.Tensor,
    output: cute.Tensor,
    rows: Int32,
    width: Int32,
    table_width: Int32,
    table_stride: Int64,
    page_size: Int64,
    num_pages: Int64,
    stream: cuda.CUstream,
):
    _map_indexed_pages_kernel(
        indices, lengths, page_table, output, rows, width, table_width,
        table_stride, page_size, num_pages,
    ).launch(
        grid=((Int64(rows) * Int64(width) + Int64(255)) // Int64(256), 1, 1),
        block=(256, 1, 1), stream=stream,
    )


def _map_indexed_pages_args(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    storage: torch.Tensor,
    *,
    page_size: int,
    num_pages: int,
) -> tuple[torch.Tensor, tuple[object, ...]]:
    rows, width = map(int, indices.shape)
    if storage.numel() < rows * width:
        raise ValueError("prepared indexed mapping storage is too small")
    output = storage.as_strided((rows, width), (width, 1))
    table_flat = (
        page_table.as_strided((page_table.shape[1],), (1,))
        if page_table.stride(0) == 0 else page_table.view(-1)
    )
    def tensor_arg(tensor: torch.Tensor):
        if hasattr(tensor, "fake_mode"):
            from cutlass.cute.runtime import make_fake_tensor

            return make_fake_tensor(
                Int32,
                (cute.sym_int(32),),
                (1,),
                assumed_align=4,
            )
        return from_dlpack(tensor, assumed_align=4).mark_layout_dynamic(
            leading_dim=0
        )

    tensors = (indices.view(-1), lengths, table_flat, output.view(-1))
    args = tuple(tensor_arg(tensor) for tensor in tensors) + (
        Int32(rows),
        Int32(width),
        Int32(page_table.shape[1]),
        Int64(page_table.stride(0)),
        Int64(page_size),
        Int64(num_pages),
        current_cuda_stream(),
    )
    return output, args


def map_indexed_pages(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    storage: torch.Tensor,
    *,
    page_size: int,
    num_pages: int,
    launcher: object | None = None,
) -> torch.Tensor:
    """Map logical indexed slots into prepared storage; invalid entries are -1."""
    output, args = _map_indexed_pages_args(
        indices, lengths, page_table, storage,
        page_size=page_size, num_pages=num_pages,
    )
    if output.numel() == 0:
        return output
    spec = KernelCompileSpec.from_key(
        "attention.compressed_sparse_mla.map_indexed_pages", 1, ()
    )
    if launcher is None:
        launch(
            _map_indexed_pages_launch,
            compile_args=args,
            runtime_args=args,
            compile_spec=spec,
        )
    else:
        from b12x._lib.compiler import run_compiled

        run_compiled(launcher, args)
    return output
