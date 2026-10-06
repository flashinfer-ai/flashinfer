"""Check shared source partitioning against independent per-byte routing."""

from collections import Counter
import random

import pytest

from b12x.loader._shared_read_plan import Copy, plan_reads


def _writes(copies):
    result = {}
    for copy in copies:
        for row in range(copy.rows):
            for column in range(copy.width):
                destination = copy.destination + row * copy.destination_stride + column * (1 + copy.expand)
                result[destination] = (copy.file, copy.offset + row * copy.source_stride + column)
    return result


@pytest.mark.parametrize("seed", range(16))
def test_random_rows_preserve_exact_routing_and_read_each_block_once(seed):
    rng = random.Random(seed)
    copies = []
    for index in range(16):
        width, rows = rng.randrange(1, 400), rng.randrange(1, 13)
        copies.append(Copy(rng.randrange(2), rng.randrange(0, 20000), width,
                           (1 << 34) + index * 65536, 0, rows,
                           rng.choice([0, rng.randrange(1, 500)]), width + 17))
    expected = _writes(copies)
    actual, reads = {}, Counter()
    for rank in range(4):
        chunks, fragments = plan_reads(copies, rank=rank, world_size=4, chunk_bytes=4096)
        for index in range(0, len(chunks), 5):
            file, offset, size, first, count = chunks[index:index + 5]
            reads.update((file, block) for block in range(offset, offset + size, 4096))
            for fragment in range(first, first + count):
                source, destination, width, rows, source_stride, destination_stride, expand = fragments[fragment * 7:fragment * 7 + 7]
                assert expand == 0
                for row in range(rows):
                    for column in range(width):
                        pointer = destination + row * destination_stride + column
                        assert pointer not in actual
                        actual[pointer] = (file, offset + source + row * source_stride + column)
    assert actual == expected
    assert set(reads.values()) == {1}


def test_bf16_fragment_offsets_and_repeated_source_rows():
    copies = [Copy(7, (1 << 33) + 4094, 8, (1 << 35) + 12, 1, 4, 0, 32)]
    expected = _writes(copies)
    actual = {}
    for rank in range(4):
        chunks, fragments = plan_reads(copies, rank=rank, world_size=4, chunk_bytes=4096)
        for index in range(0, len(chunks), 5):
            file, offset, _, first, count = chunks[index:index + 5]
            for fragment in range(first, first + count):
                source, destination, width, rows, source_stride, destination_stride, expand = fragments[fragment * 7:fragment * 7 + 7]
                assert expand == 1 and width % 2 == 0
                for row in range(rows):
                    for column in range(width):
                        actual[destination + row * destination_stride + column * 2] = (file, offset + source + row * source_stride + column)
    assert actual == expected


def test_signed_offset_overflow_and_overlapping_destinations_fail_closed():
    with pytest.raises(ValueError, match="64-bit"):
        plan_reads([Copy(0, (1 << 63) - 8, 16, 4096, 0, 1, 0, 0)], rank=0, world_size=1)
    with pytest.raises(ValueError, match="overlapping"):
        plan_reads([Copy(0, 0, 16, 4096, 0, 2, 32, 8)], rank=0, world_size=1)


def test_empty_participant_has_a_valid_empty_schedule():
    assert all(not buffer for buffer in plan_reads([], rank=3, world_size=4))
