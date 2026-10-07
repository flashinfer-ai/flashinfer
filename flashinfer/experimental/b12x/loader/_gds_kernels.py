"""Bounded byte copies and bit-preserving BF16 expansion for checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["row_bytes", "rows", "source_stride", "destination_stride"]
)
def _copy_rows(
    source,
    destination,
    row_bytes,
    rows,
    source_stride,
    destination_stride,
    EXPAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    width = row_bytes // (2 if EXPAND else 1)
    row, column = index // width, index % width
    valid = row < rows
    src = row * source_stride + column * (2 if EXPAND else 1)
    dst = row * destination_stride + column * (4 if EXPAND else 1)
    low = tl.load(source + src, valid, 0)
    if EXPAND:
        high = tl.load(source + src + 1, valid, 0)
        tl.store(destination + dst, 0, valid)
        tl.store(destination + dst + 1, 0, valid)
        tl.store(destination + dst + 2, low, valid)
        tl.store(destination + dst + 3, high, valid)
    else:
        tl.store(destination + dst, low, valid)


@dataclass(frozen=True)
class _BytePointer:
    dtype: torch.dtype = torch.uint8

    def data_ptr(self):
        return 1


def compile_copies(device):
    with torch.cuda.device(device):
        pointer = _BytePointer()
        # Force 64-bit runtime scalars without specializing on a live file range.
        scalar = 1 << 32
        programs = tuple(
            _copy_rows.warmup(
                pointer,
                pointer,
                scalar,
                scalar,
                scalar,
                scalar,
                EXPAND=expand,
                BLOCK=1024,
                num_warps=4,
                grid=(1,),
            )
            for expand in (False, True)
        )
        for program in programs:
            program._init_handles()
            m = program.metadata
            if (
                m.global_scratch_size
                or m.profile_scratch_size
                or m.num_ctas != 1
                or m.launch_cooperative_grid
                or m.launch_pdl
                or m.shared
                or m.num_warps * m.warp_size != 128
            ):
                raise RuntimeError(
                    "GDS checkpoint copy requires a scratch-free 128-thread CUDA launch"
                )
        return programs
