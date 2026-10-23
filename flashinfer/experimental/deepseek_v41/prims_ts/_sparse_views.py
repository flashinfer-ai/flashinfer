# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""CTA-local routing and masks over the original V4.1 combined sparse table."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from ....attention.prims_ts.kernels.mla_decode.helpers.sparse_indices import (
    SparseKVIndices,
    SparseKVLengths,
    SparseSourceTile,
)


class CombinedSparseRouteView(SparseKVIndices):
    def __init__(self, values, capacity):
        self.values = values
        self.indices, self.row, self.sl, self.cl, self.sc, self.cc = values
        self.capacity = capacity
        self.shape = (capacity, self.indices.shape[0])

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return CombinedSparseRouteView(
            cutlass.new_from_mlir_values(self.values, values), self.capacity
        )

    @cute.jit
    def bind_query_row(self, query_row, primary_length=None, extra_length=None):
        # The producer binds original metadata once for this nonpersistent CTA.
        del query_row, primary_length, extra_length
        return self

    @cute.jit
    def select_source_tile(
        self, token, query_row=None, *, for_mask: cutlass.Constexpr[bool] = False
    ):
        del query_row, for_mask
        compressed = token >= Int32(128)
        address = self.indices.iterator.toint() + self.row * Int64(
            self.indices.stride[0]
        ) * Int64(4)
        indices = cute.make_tensor(
            cute.make_ptr(Int32, address, assumed_align=4),
            cute.make_layout((self.indices.shape[1],), stride=(1,)),
        )
        origin = Int32(128) if compressed else Int32(0)
        length = self.cl if compressed else self.sl
        capacity = self.cc if compressed else self.sc
        tag = Int32(-2147483648) if compressed else Int32(0)
        return CombinedSparseTile((indices, origin, length, capacity, tag))

    @cute.jit
    def is_invalid(self, token, request):
        return self.select_source_tile(token, request, for_mask=True).is_invalid(
            token, request
        )

    @cute.jit
    def __getitem__(self, coordinate):
        token, request = coordinate
        return self.select_source_tile(token, request).load_row(token)


class CombinedSparseTile(SparseSourceTile):
    """Identical bounded loads and physical-ID validity for gather and softmax."""

    def __init__(self, values):
        self.values = values
        self.indices, self.origin, self.length, self.capacity, self.tag = values
        # Original metadata may contain holes, sentinels and out-of-pool IDs.
        self.valid_prefix = False

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return CombinedSparseTile(cutlass.new_from_mlir_values(self.values, values))

    @cute.jit
    def _valid_id(self, index):
        # Validate the signed original ID before source tagging. Otherwise an
        # INT_MIN hole could become main row0, or INT_MAX a valid huge-pool row.
        return (
            (index >= Int32(0))
            & (index != Int32(0x7FFFFFFF))
            & (Int64(index) < self.capacity)
        )

    @cute.jit
    def _bounded_index(self, token):
        position = token - self.origin
        in_prefix = (position >= Int32(0)) & (position < self.length)
        index = Int32(0x7FFFFFFF)
        # Guard the table read itself. K0 still schedules a masked main LOOP,
        # but its combined table has no column128 and needs no dummy indices.
        if in_prefix:
            index = Int32(self.indices[token])
        return index, in_prefix & self._valid_id(index)

    @cute.jit
    def is_invalid(self, token, request):
        del request
        _, valid = self._bounded_index(token)
        return ~valid

    @cute.jit
    def load_row(self, token):
        index, valid = self._bounded_index(token)
        return (index if valid else Int32(0x7FFFFFFF)) | self.tag

    @cute.jit
    def load_full_quad(self, token):
        # Main's staging fast path still needs original signed-ID validation.
        return self.load_quad(token)

    @cute.jit
    def load_quad(self, token):
        rows = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(4):
            rows[j] = Int32(0x7FFFFFFF) | self.tag
        position = token - self.origin
        if position >= Int32(0) and position + Int32(3) < self.length:
            values = cutlass.Pointer(self.indices.iterator + token, dtype=Int32).load(
                count=4, alignment=4
            )
            for j in cutlass.range_constexpr(4):
                index = Int32(values[j])
                rows[j] = (
                    index if self._valid_id(index) else Int32(0x7FFFFFFF)
                ) | self.tag
        elif position + Int32(3) >= Int32(0) and position < self.length:
            for j in cutlass.range_constexpr(4):
                rows[j] = self.load_row(token + Int32(j))
        return rows


@cute.jit
def bind_combined_sparse_views(
    indices, seq_lens, topk_lens, pools, capacity, request=None
):
    if cutlass.const_expr(request is None):
        raise ValueError("V4.1 combined metadata requires a nonpersistent request")
    compressed_k = indices.shape[1] - 128
    swa_length = cute.math.min(
        cute.math.max(Int32(seq_lens[request]), Int32(0)), Int32(128)
    )
    # Clamp before subtracting128 so malformed INT_MIN never wraps around.
    main_length = cute.math.min(
        cute.math.max(Int32(topk_lens[request]), Int32(128)),
        Int32(128 + compressed_k),
    ) - Int32(128)
    routes = CombinedSparseRouteView(
        (
            indices,
            request,
            swa_length,
            main_length,
            Int64(cute.size(pools[1])) // Int64(512),
            Int64(cute.size(pools[0])) // Int64(288),
        ),
        capacity,
    )
    # Fixed SWA HEAD/main LOOP origins must not depend on valid counts. The
    # existing K0 capacity256 schedule retains its fully masked second task.
    extent = Int32(128 + ((compressed_k + 127) // 128) * 128)
    return routes, SparseKVLengths((None, None, extent, indices.shape[0]))
