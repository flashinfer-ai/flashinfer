# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read separate source indices through the existing MLA loader interface.

The loaders and scheduler consume one logical KV axis. These CuTe objects
select a caller-owned primary/extra list and its live length without allocating
or writing a merged route tensor. They contain only tensor handles and scalar
SSA values; the MLIR methods preserve those values across staged task calls.
Source spans are 128-aligned so a gather tile never crosses between pools.
Premerged routes bypass these adapters; page conversion stays in preparation.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64


class SparseRouteView:
    """One request's two source lists, bound once for nonpersistent CTAs."""

    def __init__(self, values, capacity, assume_valid_prefix=False):
        self.values = values
        self.capacity = capacity
        self.assume_valid_prefix = assume_valid_prefix
        (
            self.primary_indices,
            self.extra_indices,
            self.primary_length,
            self.extra_length,
            self.request,
        ) = values
        self.primary_span = (
            (self.primary_length + Int32(127)) // Int32(128) * Int32(128)
        )
        self.shape = (capacity, self.primary_indices.shape[0])

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseRouteView(
            cutlass.new_from_mlir_values(self.values, values),
            self.capacity,
            self.assume_valid_prefix,
        )

    @cute.jit
    def __getitem__(self, coordinate):
        token, _ = coordinate
        result = Int32(0x7FFFFFFF)
        if token < self.primary_length:
            index = Int32(self.primary_indices[self.request, token])
            if index >= 0:
                result = index
        elif token >= self.primary_span:
            result = Int32(-1)
            position = token - self.primary_span
            if position < self.extra_length:
                index = Int32(self.extra_indices[self.request, position])
                if index >= 0:
                    result = index | Int32(-2147483648)
        return result

    @cute.jit
    def routes_for_tile(self, token, request=None):
        use_extra = token >= self.primary_span
        primary_ptr = self.primary_indices.iterator.toint() + self.request * Int64(
            self.primary_indices.stride[0]
        ) * Int64(4)
        extra_ptr = self.extra_indices.iterator.toint() + self.request * Int64(
            self.extra_indices.stride[0]
        ) * Int64(4)
        address = extra_ptr if use_extra else primary_ptr
        origin = self.primary_span if use_extra else Int32(0)
        length = self.extra_length if use_extra else self.primary_length
        tag = Int32(-2147483648) if use_extra else Int32(0)
        indices = cute.make_tensor(
            cute.make_ptr(Int32, address, assumed_align=4),
            cute.make_layout((self.capacity,), stride=(1,)),
        )
        return SparseTileRoutes((indices, origin, length, tag))

    @cute.jit
    def mask_for_tile(self, token, request=None):
        # Source boundaries are 128-row aligned. Select the pointer and prefix
        # once per score tile, outside the unrolled register mask loop.
        use_extra = token >= self.primary_span
        if cutlass.const_expr(self.assume_valid_prefix):
            origin = self.primary_span if use_extra else Int32(0)
            length = self.extra_length if use_extra else self.primary_length
            return SparsePrefixMask((origin, length))
        routes = self.routes_for_tile(token)
        return routes

    @cute.jit
    def is_invalid(self, token, request):
        return self.mask_for_tile(token).is_invalid(token, request)


class SparseLengthView:
    """Expose the already-loaded request length to the shared scheduler."""

    def __init__(self, values):
        self.values = values
        self.length, rows = values
        self.shape = (rows,)

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseLengthView(cutlass.new_from_mlir_values(self.values, values))

    @cute.jit
    def __getitem__(self, request):
        del request
        return self.length


class SparsePrefixMask:
    """Analytic validity for one source's compact, hole-free prefix."""

    def __init__(self, values):
        self.values = values
        self.origin, self.length = values

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparsePrefixMask(cutlass.new_from_mlir_values(self.values, values))

    @cute.jit
    def tile_is_full(self, start, live_length):
        return start + Int32(128) <= cute.math.min(
            self.origin + self.length, live_length
        )

    @cute.jit
    def prefix_mask32(self, start, live_length):
        # Tiles and source boundaries are 128-aligned; start never precedes
        # this source's origin. Include the request's live bound as well.
        end = cute.math.min(self.origin + self.length, live_length)
        valid = cute.math.min(cute.math.max(end - start, Int32(0)), Int32(32))
        # Mask the shift count even on the unselected side of the expression.
        # A full 32-bit valid prefix must not execute an undefined shift by 32.
        bits = Int32(-1) << (valid & Int32(31))
        return bits if valid < Int32(32) else Int32(0)

    @cute.jit
    def is_invalid(self, token, request):
        del request
        return (token < self.origin) | (token >= self.origin + self.length)


class SparseTileRoutes:
    """One source selection per tile, with bounded scalar and quad loads."""

    def __init__(self, values):
        self.values = values
        self.indices, self.origin, self.length, self.tag = values

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseTileRoutes(cutlass.new_from_mlir_values(self.values, values))

    @cute.jit
    def _bounded_index(self, token):
        position = token - self.origin
        # Zero-capacity sources bind a one-row dummy. Clamp even masked
        # positions so both gather and softmax can use a bounded load.
        safe = cute.math.min(
            cute.math.max(position, Int32(0)),
            cute.math.max(self.length - Int32(1), Int32(0)),
        )
        index = Int32(self.indices[safe])
        valid = (position >= Int32(0)) & (position < self.length) & (index >= Int32(0))
        return index, valid

    @cute.jit
    def is_invalid(self, token, request):
        del request
        _, valid = self._bounded_index(token)
        return ~valid

    @cute.jit
    def mapped_row(self, token):
        index, valid = self._bounded_index(token)
        return (index if valid else Int32(0x7FFFFFFF)) | self.tag

    @cute.jit
    def mapped_quad(self, token):
        """Load a complete active quad together and preserve partial-prefix masking."""
        rows = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(4):
            rows[j] = Int32(0x7FFFFFFF) | self.tag
        position = token - self.origin
        if position >= Int32(0) and position + Int32(3) < self.length:
            values = cutlass.Pointer(
                self.indices.iterator + position, dtype=Int32
            ).load(count=4, alignment=4)
            for j in cutlass.range_constexpr(4):
                index = Int32(values[j])
                rows[j] = (index if index >= Int32(0) else Int32(0x7FFFFFFF)) | self.tag
        elif position + Int32(3) >= Int32(0) and position < self.length:
            for j in cutlass.range_constexpr(4):
                rows[j] = self.mapped_row(token + Int32(j))
        return rows


class SparseBatchRouteView:
    """Bind live source metadata to the request currently owned by a work tile."""

    def __init__(self, values, capacity, assume_valid_prefix=False):
        self.values = values
        (
            self.primary_indices,
            self.extra_indices,
            self.primary_lengths,
            self.extra_lengths,
        ) = values
        self.capacity = capacity
        self.assume_valid_prefix = assume_valid_prefix
        self.shape = (capacity, self.primary_indices.shape[0])

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseBatchRouteView(
            cutlass.new_from_mlir_values(self.values, values),
            self.capacity,
            self.assume_valid_prefix,
        )

    @cute.jit
    def for_request(self, request, primary_length=None, extra_length=None):
        # Persistent consumers may reuse lengths loaded at query-tile entry.
        row = Int64(request)
        return SparseRouteView(
            (
                self.primary_indices,
                self.extra_indices,
                Int32(self.primary_lengths[row])
                if cutlass.const_expr(primary_length is None)
                else primary_length,
                Int32(self.extra_lengths[row])
                if cutlass.const_expr(extra_length is None)
                else extra_length,
                row,
            ),
            self.capacity,
            self.assume_valid_prefix,
        )

    @cute.jit
    def __getitem__(self, coordinate):
        return self.for_request(coordinate[1])[coordinate]

    @cute.jit
    def routes_for_tile(self, token, request):
        return self.for_request(request).routes_for_tile(token)

    @cute.jit
    def mask_for_tile(self, token, request):
        return self.for_request(request).mask_for_tile(token)

    @cute.jit
    def is_invalid(self, token, request):
        return self.mask_for_tile(token, request).is_invalid(token, request)


class SparseBatchLengthView:
    """Live padded source extent for each virtual request."""

    def __init__(self, values):
        self.values = values
        self.primary_lengths, self.extra_lengths = values
        self.shape = self.primary_lengths.shape

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseBatchLengthView(cutlass.new_from_mlir_values(self.values, values))

    @cute.jit
    def __getitem__(self, request):
        return _sparse_execution_length(
            Int32(self.primary_lengths[request]), Int32(self.extra_lengths[request])
        )


@cute.jit
def _sparse_execution_length(primary_length, extra_length):
    primary_tiles = (primary_length + Int32(127)) // Int32(128)
    extra_tiles = (extra_length + Int32(127)) // Int32(128)
    # Empty requests still execute a masked tile to publish O=0 and LSE=-inf.
    return cute.math.max((primary_tiles + extra_tiles) * Int32(128), Int32(1))


@cute.jit
def bind_sparse_views(
    si,
    ci,
    sl,
    cl,
    capacity: cutlass.Constexpr[int],
    *,
    request=None,
    assume_valid_prefix: cutlass.Constexpr[bool] = False,
):
    """Bind source lists to a work queue or to one nonpersistent request."""
    routes = SparseBatchRouteView((si, ci, sl, cl), capacity, assume_valid_prefix)
    if cutlass.const_expr(request is None):
        return routes, SparseBatchLengthView((sl, cl))
    else:
        current = routes.for_request(request)
        length = _sparse_execution_length(current.primary_length, current.extra_length)
        return current, SparseLengthView((length, si.shape[0]))
