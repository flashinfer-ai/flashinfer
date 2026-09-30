"""Partition routed checkpoint rows into bounded, disjoint shared reads."""

from __future__ import annotations

from array import array
from dataclasses import dataclass


@dataclass(frozen=True)
class Copy:
    file: int
    offset: int
    width: int
    destination: int
    expand: int
    rows: int
    source_stride: int
    destination_stride: int

    @property
    def end(self):
        return self.offset + (self.rows - 1) * self.source_stride + self.width


def _fragments(copy, begin, end):
    stride = copy.source_stride
    if stride:
        first = max(0, (begin - copy.offset - copy.width) // stride + 1)
        last = min(copy.rows - 1, (end - 1 - copy.offset) // stride)
        full_first = max(first, -((copy.offset - begin) // stride))
        full_last = min(last, (end - copy.offset - copy.width) // stride)
    else:
        first, last = 0, copy.rows - 1
        full_first, full_last = first, last

    def fragment(row, count):
        start = copy.offset + row * stride
        low, high = max(start, begin), min(start + copy.width, end)
        if high <= low:
            return
        if copy.expand and ((low - start) % 2 or (high - low) % 2):
            raise ValueError("shared BF16 reads must preserve element boundaries")
        yield (
            low - begin,
            copy.destination + row * copy.destination_stride
            + (low - start) * (1 + copy.expand),
            high - low, count, stride, copy.destination_stride, copy.expand,
        )

    if full_first <= full_last:
        for row in range(first, full_first):
            yield from fragment(row, 1)
        yield from fragment(full_first, full_last - full_first + 1)
        for row in range(full_last + 1, last + 1):
            yield from fragment(row, 1)
    else:
        for row in range(first, last + 1):
            yield from fragment(row, 1)


def plan_reads(copies, *, rank, world_size, chunk_bytes=4653056):
    """Return native chunk/fragment arrays for one owner rank.

    Source envelopes include strided gaps. Their aligned union is read once
    across the group; only routed row bytes are written to destinations.
    Destination pointers must already refer to mappings on the owner's device.
    """
    if not 0 <= rank < world_size or chunk_bytes <= 0 or chunk_bytes % 4096:
        raise ValueError("invalid shared read rank or aligned chunk capacity")
    ordered = sorted(copies, key=lambda c: (c.file, c.offset))
    for copy in ordered:
        values = (copy.file, copy.offset, copy.width, copy.destination,
                  copy.source_stride, copy.destination_stride)
        if any(v < 0 or v > (1 << 63) - 1 for v in values):
            raise ValueError("shared read descriptor exceeds signed 64-bit bounds")
        if copy.rows <= 0 or copy.width <= 0 or copy.expand not in (0, 1):
            raise ValueError("invalid shared read row geometry")
        if (copy.end > (1 << 63) - 1
                or copy.destination + (copy.rows - 1) * copy.destination_stride
                + copy.width * (1 + copy.expand) > (1 << 63) - 1):
            raise ValueError("shared read address arithmetic exceeds signed 64-bit bounds")
        if copy.rows > 1 and copy.destination_stride < copy.width * (1 + copy.expand):
            raise ValueError("overlapping shared read destination rows")
    chunks, fragments = array("Q"), array("Q")
    sequence = start = 0
    while start < len(ordered):
        stop = start + 1
        file = ordered[start].file
        begin = ordered[start].offset // 4096 * 4096
        end = (ordered[start].end + 4095) // 4096 * 4096
        while stop < len(ordered) and ordered[stop].file == file and ordered[stop].offset <= end:
            end = max(end, (ordered[stop].end + 4095) // 4096 * 4096)
            stop += 1
        active, next_copy = [], start
        for offset in range(begin, end, chunk_bytes):
            limit = min(offset + chunk_bytes, end)
            active = [copy for copy in active if copy.end > offset]
            while next_copy < stop and ordered[next_copy].offset < limit:
                active.append(ordered[next_copy])
                next_copy += 1
            if sequence % world_size == rank:
                first = len(fragments) // 7
                for copy in active:
                    for fragment in _fragments(copy, offset, limit):
                        fragments.extend(fragment)
                chunks.extend((file, offset, limit - offset, first, len(fragments) // 7 - first))
            sequence += 1
        start = stop
    return chunks, fragments
