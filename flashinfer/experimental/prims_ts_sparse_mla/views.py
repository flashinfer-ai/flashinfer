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

"""CTA-local views of live sparse metadata, without a preparation launch.

One nonpersistent CTA owns one virtual query request. Prefix lengths are
loaded once; row mapping is evaluated when the page-loader consumes indices.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64


class SparseRouteView:
    def __init__(self, values, pages, capacity, assume_valid_prefix=False):
        self.values = values
        self.pages = pages
        self.capacity = capacity
        self.assume_valid_prefix = assume_valid_prefix
        self.si, self.ci, self.sl, self.cl, self.row, self.ss, self.cs = values
        self.swa_span = (self.sl + Int32(127)) // Int32(128) * Int32(128)
        self.shape = (capacity, self.si.shape[0])

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseRouteView(
            cutlass.new_from_mlir_values(self.values, values),
            self.pages,
            self.capacity,
            self.assume_valid_prefix,
        )

    @cute.jit
    def __getitem__(self, coordinate):
        token, _ = coordinate
        result = Int32(0x7FFFFFFF)
        if token < self.sl:
            index = Int32(self.si[self.row, token])
            if index >= 0:
                result = index // self.pages[0] * self.ss + index % self.pages[0]
        elif token >= self.swa_span:
            result = Int32(-1)
            position = token - self.swa_span
            if position < self.cl:
                index = Int32(self.ci[self.row, position])
                if index >= 0:
                    result = (
                        index // self.pages[1] * self.cs + index % self.pages[1]
                    ) | Int32(-2147483648)
        return result

    @cute.jit
    def routes_for_tile(self, token, request=None):
        compressed = token >= self.swa_span
        primary_ptr = self.si.iterator.toint() + self.row * Int64(
            self.si.stride[0]
        ) * Int64(4)
        extra_ptr = self.ci.iterator.toint() + self.row * Int64(
            self.ci.stride[0]
        ) * Int64(4)
        address = extra_ptr if compressed else primary_ptr
        origin = self.swa_span if compressed else Int32(0)
        length = self.cl if compressed else self.sl
        shift = (
            Int32(self.pages[1].bit_length() - 1)
            if compressed
            else Int32(self.pages[0].bit_length() - 1)
        )
        stride = self.cs if compressed else self.ss
        tag = Int32(-2147483648) if compressed else Int32(0)
        indices = cute.make_tensor(
            cute.make_ptr(Int32, address, assumed_align=4),
            cute.make_layout((self.capacity,), stride=(1,)),
        )
        return SparseTileRoutes((indices, origin, length, shift, stride, tag))

    @cute.jit
    def mask_for_tile(self, token, request=None):
        # Source boundaries are 128-row aligned. Select the pointer and prefix
        # once per score tile, outside the unrolled register mask loop.
        compressed = token >= self.swa_span
        if cutlass.const_expr(self.assume_valid_prefix):
            origin = self.swa_span if compressed else Int32(0)
            length = self.cl if compressed else self.sl
            return SparsePrefixMask((origin, length))
        primary_ptr = self.si.iterator.toint() + self.row * Int64(
            self.si.stride[0]
        ) * Int64(4)
        extra_ptr = self.ci.iterator.toint() + self.row * Int64(
            self.ci.stride[0]
        ) * Int64(4)
        address = extra_ptr if compressed else primary_ptr
        origin = self.swa_span if compressed else Int32(0)
        length = self.cl if compressed else self.sl
        indices = cute.make_tensor(
            cute.make_ptr(Int32, address, assumed_align=4),
            cute.make_layout((self.capacity,), stride=(1,)),
        )
        return SparseTileMask((indices, origin, length))

    @cute.jit
    def is_invalid(self, token, request):
        return self.mask_for_tile(token).is_invalid(token, request)


class SparseLengthView:
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


class SparseTileMask:
    def __init__(self, values):
        self.values = values
        self.indices, self.origin, self.length = values

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseTileMask(cutlass.new_from_mlir_values(self.values, values))

    @cute.jit
    def is_invalid(self, token, request):
        del request
        position = token - self.origin
        # A clamped load keeps the mask branchless without reading beyond a
        # source's active prefix. Zero-capacity sources bind a one-row dummy.
        safe = cute.math.min(
            cute.math.max(position, Int32(0)),
            cute.math.max(self.length - Int32(1), Int32(0)),
        )
        index = Int32(self.indices[safe])
        return (position < Int32(0)) | (position >= self.length) | (index < Int32(0))


class SparseTileRoutes:
    """One source selection per tile, with bounded scalar and quad loads."""

    def __init__(self, values):
        self.values = values
        self.indices, self.origin, self.length, self.shift, self.stride, self.tag = (
            values
        )

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseTileRoutes(cutlass.new_from_mlir_values(self.values, values))

    @cute.jit
    def mapped_row(self, token):
        position = token - self.origin
        safe = cute.math.min(
            cute.math.max(position, Int32(0)),
            cute.math.max(self.length - Int32(1), Int32(0)),
        )
        index = Int32(self.indices[safe])
        valid = (position >= Int32(0)) & (position < self.length) & (index >= Int32(0))
        mapped = (index >> self.shift) * self.stride + (
            index & ((Int32(1) << self.shift) - Int32(1))
        )
        return (mapped if valid else Int32(0x7FFFFFFF)) | self.tag

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
            if self.stride == (Int32(1) << self.shift):
                for j in cutlass.range_constexpr(4):
                    index = Int32(values[j])
                    rows[j] = (
                        index if index >= Int32(0) else Int32(0x7FFFFFFF)
                    ) | self.tag
            else:
                for j in cutlass.range_constexpr(4):
                    index = Int32(values[j])
                    mapped = (index >> self.shift) * self.stride + (
                        index & ((Int32(1) << self.shift) - Int32(1))
                    )
                    rows[j] = (
                        mapped if index >= Int32(0) else Int32(0x7FFFFFFF)
                    ) | self.tag
        elif position + Int32(3) >= Int32(0) and position < self.length:
            for j in cutlass.range_constexpr(4):
                rows[j] = self.mapped_row(token + Int32(j))
        return rows


class SparseBatchRouteView:
    """Bind live source metadata to the request currently owned by a work tile."""

    def __init__(self, values, pages, capacity, assume_valid_prefix=False):
        self.values = values
        self.si, self.ci, self.sl, self.cl, self.ss, self.cs = values
        self.pages = pages
        self.capacity = capacity
        self.assume_valid_prefix = assume_valid_prefix
        self.shape = (capacity, self.si.shape[0])

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseBatchRouteView(
            cutlass.new_from_mlir_values(self.values, values),
            self.pages,
            self.capacity,
            self.assume_valid_prefix,
        )

    @cute.jit
    def for_request(self, request):
        row = Int64(request)
        return SparseRouteView(
            (
                self.si,
                self.ci,
                Int32(self.sl[row]),
                Int32(self.cl[row]),
                row,
                self.ss,
                self.cs,
            ),
            self.pages,
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
        self.sl, self.cl = values
        self.shape = self.sl.shape

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseBatchLengthView(cutlass.new_from_mlir_values(self.values, values))

    @cute.jit
    def __getitem__(self, request):
        swa = (Int32(self.sl[request]) + Int32(127)) // Int32(128)
        compressed = (Int32(self.cl[request]) + Int32(127)) // Int32(128)
        # Keep one masked tile for an empty request so fused output publishes
        # the sink-only result, matching the existing nonpersistent route.
        return cute.math.max((swa + compressed) * Int32(128), Int32(1))
