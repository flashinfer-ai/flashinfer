"""Dense page-table traversal and asynchronous K3 record staging."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    shared_ptr_to_u32,
)

from ._layout import CANDIDATES_PER_CHUNK


@cute.jit
def issue_dense_page_gather(
    cache_bytes: cute.Tensor,
    page_table: cute.Tensor,
    kv_dst_addr: Int32,
    full_mbar_ptr,
    request: Int32,
    token_begin: Int32,
    token_end: Int32,
    io_lane: Int32,
    page_stride_bytes: Int64,
    record_stride_bytes_global: Int64,
    page_table_stride: Int64,
    *,
    page_size: cutlass.Constexpr,
    record_bytes: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr,
):
    """Stage one 64-record dense window with an mbarrier transaction.

    Every pool-scaled product is promoted before multiplication. Invalid tail
    rows copy physical record zero and are masked by the consumer; issuing a
    fixed byte count keeps the full-barrier protocol independent of live
    sequence length and therefore graph-replay stable.
    """
    full_mbar_addr = shared_ptr_to_u32(full_mbar_ptr)
    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_mbar_ptr,
            Int32(CANDIDATES_PER_CHUNK * record_bytes),
        )

    entry = io_lane
    for _ in cutlass.range_constexpr(CANDIDATES_PER_CHUNK // 32):
        logical_token = token_begin + entry
        safe_token = logical_token
        if logical_token >= token_end:
            safe_token = Int32(0)
        logical_page = safe_token // Int32(page_size)
        in_page = safe_token - logical_page * Int32(page_size)
        physical_page = Int32(
            page_table[request.to(Int64) * page_table_stride + logical_page.to(Int64)]
        )

        # Load-bearing Int64 conversions. Neither multiplication is allowed to
        # occur in Int32, even when benchmark page ids happen to be small.
        source_offset = (
            physical_page.to(Int64) * page_stride_bytes
            + in_page.to(Int64) * record_stride_bytes_global
        )
        cp_async_bulk_g2s_mbar(
            kv_dst_addr + entry * Int32(record_stride_bytes),
            get_ptr_as_int64(cache_bytes, source_offset),
            Int32(record_bytes),
            full_mbar_addr,
        )
        entry += Int32(32)


__all__ = ["issue_dense_page_gather"]
