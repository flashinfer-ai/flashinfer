# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared masking predicates for PrimTS attention kernels.

Every PrimTS softmax follows one masking design::

    once per work tile:   [begin, end) = unmasked_kv_tile_range(...)
    once per KV tile:     masked = not kv_tile_in_range(tile_idx, begin, end)
                          (or packed_kv_tile_in_range on one packed word)

      KV tiles   0 .. begin-1 | begin ........ end-1 | end .. N-1
                 left-masked  |  unmasked (no mask)  | right-masked

The range is the intersection of every row's visible key interval in the
work tile, rounded inward to whole tiles, in absolute KV tile indices. It
is computed once, before the KV loop, from the work tile's geometry
(sequence lengths, causal offset, sliding window, explicit K bounds), so a
KV tile pays one range test instead of rebuilding row bounds. A kernel
whose single right bound is already live in the loop compares it in key
space instead (``kv_tile_needs_right_mask``), which needs no extra
loop-carried state. Masked tiles
keep their exact per-row/per-key mask code; unmasked tiles take a
specialization without it. Data-dependent sparse masks (sparse routes,
page-membership words) are outside this range test and keep their own
predicates.

All helpers accept Python integers (host-side tests, trace-time constants)
or CuTe DSL integer values (device code). Intervals are half-open.
"""


def _imax(a, b):
    """Return ``max(a, b)`` for Python or CuTe DSL integers."""
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    import cutlass.cute as cute
    from cutlass import Int32

    return cute.math.max(Int32(a), Int32(b))


def _imin(a, b):
    """Return ``min(a, b)`` for Python or CuTe DSL integers."""
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    import cutlass.cute as cute
    from cutlass import Int32

    return cute.math.min(Int32(a), Int32(b))


def unmasked_kv_tile_range(visible_k_begin, visible_k_end, tile_size_kv):
    """Return the absolute KV tile range ``[begin, end)`` that needs no mask.

    ``visible_k_begin`` is the largest left bound and ``visible_k_end`` the
    smallest right bound over every row of the work tile, so a tile inside
    ``[visible_k_begin, visible_k_end)`` is visible to all rows. ``begin``
    rounds the left bound up and ``end`` rounds the right bound down to a
    tile edge; an empty intersection yields ``begin == end`` (every tile is
    masked).
    """
    begin_k = _imax(visible_k_begin, 0)
    end_k = _imax(visible_k_end, 0)
    begin = (begin_k + (tile_size_kv - 1)) // tile_size_kv
    end = _imax(end_k // tile_size_kv, begin)
    return begin, end


def kv_tile_in_range(tile_idx, begin, end):
    """Return whether absolute KV tile ``tile_idx`` lies in ``[begin, end)``.

    With ``[begin, end)`` from :func:`unmasked_kv_tile_range`, true selects
    the unmasked specialization.
    """
    return (tile_idx >= begin) & (tile_idx < end)


def kv_tile_needs_right_mask(tile_offset_k, tile_size_kv, visible_k_end):
    """Return whether a KV tile crosses a row-visible right bound.

    The key-space form of the range test, for kernels whose right bound is
    already live in the KV loop: comparing it against the tile's first key
    adds no loop-carried register, where a cached tile range would.
    """
    return tile_offset_k + tile_size_kv > visible_k_end


def kv_tile_is_fully_visible(
    tile_offset_k,
    tile_size_kv,
    visible_k_begin,
    visible_k_end,
):
    """Return whether a KV tile is inside every row's visible interval.

    The key-space form of the range test for a two-sided interval whose
    bounds are already live in the KV loop. All intervals are half-open.
    """
    return (tile_offset_k >= visible_k_begin) & (
        tile_offset_k + tile_size_kv <= visible_k_end
    )


_PACKED_TILE_MAX = 0xFFFF


def pack_kv_tile_range(begin, end):
    """Pack a tile range ``[begin, end)`` into one 32-bit word, ``end << 16 | begin``.

    Kernels whose softmax registers are at their budget carry the range
    through the KV loop as one live value instead of two. Both bounds are
    clamped to 65535, which only shrinks the range: an open or far right bound
    keeps tiles from 65535 on masked, and a range starting there is empty.
    """
    begin = _imin(begin, _PACKED_TILE_MAX)
    end = _imin(end, _PACKED_TILE_MAX)
    return (end << 16) | begin


def packed_kv_tile_in_range(tile_idx, packed_range):
    """Return whether ``tile_idx`` lies in a range packed by :func:`pack_kv_tile_range`.

    The 32-bit word may be negative, so the upper field is decoded unsigned.
    """
    return kv_tile_in_range(
        tile_idx,
        packed_range & _PACKED_TILE_MAX,
        (packed_range >> 16) & _PACKED_TILE_MAX,
    )
