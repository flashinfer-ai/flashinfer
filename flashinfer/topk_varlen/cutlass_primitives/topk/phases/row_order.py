"""Row order for wide ragged batches: a one-CTA prepare launch that ranks rows longest-first.

A batch wider than the SM count puts two (or more) CTAs on an SM, and its time is the time of
the slowest pair.  The hardware scheduler assigns CTAs in launch order, so with random lengths
the long rows land anywhere; launched longest-first, the first wave holds the long rows and the
short rows paired beside them leave early, so the long ones finish alone (B200 64K b=256 k=2048
ragged, rows sorted on the host: 16.9 -> 14.6 us).  Ranking inside the main kernel cost every
CTA a global round trip (docs/measured-worse.md, lpt_order); this kernel ranks once, in one CTA,
before the main launch, and hands the permutation over through global memory.  With
programmatic dependent launch the main kernel is resident and waiting while this one runs.

The rank is a 256-bucket counting sort on the effective row length (longest bucket first):
one pass counts, a block scan turns the counts into descending offsets, one pass scatters.
Rows in one bucket keep an arbitrary (atomic) order, which is all the schedule needs.  The
lengths are the caller's ``lengths`` array (one entry per ``next_n`` rows), so the rank costs
one load per row and no sync: it composes with CUDA-graph capture and recomputes on replay.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

try:  # the allocator moved between DSL releases; both spellings are in use on our parts
    from cutlass.memory.smem import SmemAllocator
except ImportError:  # pragma: no cover - older nvidia-cutlass-dsl
    from cutlass.utils.smem_allocator import SmemAllocator
try:
    from cuda.bindings import driver as cuda_driver
except ImportError:  # pragma: no cover - older cuda-python
    from cuda import cuda as cuda_driver

from ...block.reduce import block_exclusive_scan_i32
from ...device.atomics import shared_add, shared_count
from ...device.launch import release_dependent_grid, wait_for_prior_grid

from .varlen import effective_length

__all__ = ["RowOrder", "rank_rows"]

BUCKETS = 256


class RowOrder:
    """One CTA writes ``order[i]`` = the row of rank ``i`` by effective length, longest first."""

    def __init__(
        self,
        n_cols: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        threads: int = 1024,
        pdl: bool = False,
    ):
        self.n_cols = n_cols
        self.next_n = next_n
        self.compress_ratio = compress_ratio
        self.threads = threads
        self.pdl = pdl

    @cute.kernel
    def kernel(self, lengths: cute.Tensor, order: cute.Tensor):
        threads = cutlass.const_expr(self.threads)
        n_cols = cutlass.const_expr(self.n_cols)
        tidx, _, _ = cute.arch.thread_idx()
        rows = cutlass.Int32(order.shape[0])

        smem = SmemAllocator()
        s_hist = smem.allocate_array(cutlass.Int32, BUCKETS, byte_alignment=128)
        s_slots = smem.allocate_array(cutlass.Int32, threads // 32, byte_alignment=128)

        if cutlass.const_expr(self.pdl):
            # the main kernel may become resident now; its own wait covers this grid's writes
            release_dependent_grid()
            wait_for_prior_grid()
        if tidx < BUCKETS:
            s_hist[tidx] = cutlass.Int32(0)
        cute.arch.barrier()
        # bucket = length scaled to 0..255 (255 = the longest rows)
        hist_base = s_hist.toint()
        for i in range(tidx, rows, threads):
            length = effective_length(
                lengths, i, n_cols, self.next_n, self.compress_ratio
            )
            b = (length * cutlass.Int32(BUCKETS - 1)) // cutlass.Int32(n_cols)
            shared_count(hist_base + b * 4)
        cute.arch.barrier()
        # descending exclusive offsets: bucket b starts after every longer bucket
        count = cutlass.Int32(0)
        if tidx < BUCKETS:
            count = s_hist[BUCKETS - 1 - tidx]
        before, _total = block_exclusive_scan_i32(count, s_slots, tidx, threads)
        cute.arch.barrier()  # every thread has read its count before the histogram is overwritten
        if tidx < BUCKETS:
            s_hist[BUCKETS - 1 - tidx] = before
        cute.arch.barrier()
        for i in range(tidx, rows, threads):
            length = effective_length(
                lengths, i, n_cols, self.next_n, self.compress_ratio
            )
            b = (length * cutlass.Int32(BUCKETS - 1)) // cutlass.Int32(n_cols)
            pos = shared_add(s_hist + b, 1)
            order[pos] = i

    @cute.jit
    def launch(
        self, lengths: cute.Tensor, order: cute.Tensor, stream: cuda_driver.CUstream
    ):
        self.kernel(lengths, order).launch(
            grid=(1, 1, 1), block=(self.threads, 1, 1), use_pdl=self.pdl, stream=stream
        )


_compiled: dict = {}


def rank_rows(
    lengths: torch.Tensor,
    order: torch.Tensor,
    n_cols: int,
    next_n: int = 1,
    compress_ratio: int = 1,
    pdl: bool = False,
) -> torch.Tensor:
    """Fill ``order`` (rows, int32) with the rows ranked longest-first by their effective
    length under ``lengths`` (rows // next_n, int32); returns ``order``.  One launch on the
    current stream."""
    assert (
        lengths.dtype == torch.int32
        and order.dtype == torch.int32
        and order.is_contiguous()
    )
    key = (
        order.shape[0],
        lengths.shape[0],
        n_cols,
        next_n,
        compress_ratio,
        pdl,
        lengths.device.index,
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(lengths.device).cuda_stream)
    args = (from_dlpack(lengths), from_dlpack(order), stream)
    if key not in _compiled:
        _compiled[key] = cute.compile(
            RowOrder(n_cols, next_n, compress_ratio, 1024, pdl).launch, *args
        )
    _compiled[key](*args)
    return order
