"""Byte-level coverage for bounded io_uring PLE transactions (no CUDA work)."""

from array import array

import pytest

from b12x.loader._native import load


@pytest.fixture
def reader_factory():
    native = load()

    def create(
        shard_rows,
        padded_rows,
        tp_start,
        tp_end,
        weight_bytes,
        scale_bytes,
        capacity,
    ):
        try:
            return native.ple_reader(
                shard_rows,
                padded_rows,
                tp_start,
                tp_end,
                weight_bytes,
                scale_bytes,
                capacity,
                4,
            )
        except RuntimeError as error:
            message = str(error)
            if (
                "support is unavailable" in message
                or "initialization failed: Operation not permitted" in message
                or "initialization failed: Function not implemented" in message
            ):
                pytest.skip(message)
            raise

    return native, create


def _touched_blocks(block_bytes, spans):
    blocks = set()
    for offset, size in spans:
        blocks.update(
            range(
                offset // block_bytes,
                (offset + size - 1) // block_bytes + 1,
            )
        )
    return blocks


def _coalesced_read_counts(blocks, block_bytes):
    """Count bounded contiguous runs without treating unrequested gaps as reads."""
    max_blocks = 65536 // block_bytes
    calls = coalesced = 0
    previous = None
    run_blocks = 0
    for block in [*sorted(blocks), None]:
        if block is None or (previous is not None and block != previous + 1):
            full_reads, remainder = divmod(run_blocks, max_blocks)
            calls += full_reads + bool(remainder)
            coalesced += full_reads + (remainder > 1)
            run_blocks = 0
        if block is not None:
            run_blocks += 1
            previous = block
    return calls, coalesced


def _expected_rows(payloads, ids, shard_rows, width):
    return b"".join(
        payloads[row // shard_rows][
            (row % shard_rows) * width : (row % shard_rows + 1) * width
        ]
        for row in ids
    )


def test_cross_block_rows_deduplicate_planes_and_preserve_compact_order(
    tmp_path, reader_factory
):
    native, create = reader_factory
    row_bytes, scale_bytes, rows = 5003, 513, 10
    weights = bytes((i * 17 + i // 257) % 256 for i in range(rows * row_bytes))
    scales = bytes((i * 31 + 5) % 256 for i in range(rows * scale_bytes))
    weight_offset = 4103
    scale_offset = weight_offset + len(weights) + 31
    path = tmp_path / "both_planes"
    path.write_bytes(b"x" * weight_offset + weights + b"x" * 31 + scales)
    reader = create(6, rows, 2, 8, row_bytes, scale_bytes, 10)
    for shard in [1, 0]:
        native.ple_reader_add(
            reader, shard, str(path), weight_offset + shard * 6 * row_bytes, False
        )
        native.ple_reader_add(
            reader, shard, str(path), scale_offset + shard * 6 * scale_bytes, True
        )
    ids = array("q", [2, 2, 5, 6, 7, 0, -1, 8, 10, 100])
    output = bytearray(b"z" * (len(ids) * row_bytes + 7))
    output_scales = bytearray(b"z" * (len(ids) * scale_bytes + 7))
    allocation = native.ple_reader_stats(reader)
    native.ple_reader_run(reader, ids, output, output_scales, len(ids))
    for i, row in enumerate(ids):
        assert output[i * row_bytes : (i + 1) * row_bytes] == (
            weights[row * row_bytes : (row + 1) * row_bytes]
            if 2 <= row < 8
            else bytes(row_bytes)
        )
        assert output_scales[i * scale_bytes : (i + 1) * scale_bytes] == (
            scales[row * scale_bytes : (row + 1) * scale_bytes]
            if 2 <= row < 8
            else bytes(scale_bytes)
        )
    assert output[-7:] == output_scales[-7:] == b"z" * 7
    stats = native.ple_reader_stats(reader)
    # The fragment floor is per source file: statx(STATX_DIOALIGN) may report a
    # 512 B logical block (ext4/NVMe) or nothing at all, which falls back to the
    # 4 KiB reference split. Take the arithmetic from the reader itself so the
    # expectations hold on either filesystem.
    block_bytes = stats["dio_align"]
    blocks = _touched_blocks(
        block_bytes,
        (
            (offset + row * size, size)
            for row in ids
            if 2 <= row < 8
            for offset, size in (
                (weight_offset, row_bytes),
                (scale_offset, scale_bytes),
            )
        ),
    )
    assert stats["unique_blocks"] == len(blocks)
    assert stats["requested_bytes"] == 5 * (row_bytes + scale_bytes)
    assert stats["read_bytes"] == sum(
        min(block_bytes, path.stat().st_size - block * block_bytes) for block in blocks
    )
    expected_calls, expected_coalesced = _coalesced_read_counts(blocks, block_bytes)
    assert stats["read_calls"] == expected_calls
    assert stats["coalesced_reads"] == expected_coalesced
    assert stats["staging_bytes"] == allocation["staging_bytes"]
    assert stats["metadata_bytes"] == allocation["metadata_bytes"]

    # The same allocations support a smaller subsequent batch without stale rows.
    second_ids = array("q", [7, 6, 2])
    native.ple_reader_run(reader, second_ids, output, output_scales, 3)
    assert output[: 3 * row_bytes] == b"".join(
        weights[row * row_bytes : (row + 1) * row_bytes] for row in second_ids
    )
    assert output_scales[: 3 * scale_bytes] == b"".join(
        scales[row * scale_bytes : (row + 1) * scale_bytes] for row in second_ids
    )
    assert native.ple_reader_stats(reader)["lookups"] == 3
    before = bytes(output)
    native.ple_reader_run(reader, array("q"), output, output_scales, 0)
    assert bytes(output) == before
    assert native.ple_reader_stats(reader)["read_calls"] == 0


def test_sparse_requests_never_read_gaps_and_cap_coalescing(tmp_path, reader_factory):
    native, create = reader_factory
    path = tmp_path / "pages"
    payload = b"".join(bytes([row]) * 4096 for row in range(40))
    path.write_bytes(payload)
    reader = create(40, 40, 0, 40, 4096, 0, 24)
    native.ple_reader_add(reader, 0, str(path), 0, False)
    ids = array("q", [*range(20), 39, 39])
    output = bytearray(len(ids) * 4096)
    native.ple_reader_run(reader, ids, output, None, len(ids))
    assert output == b"".join(payload[row * 4096 : (row + 1) * 4096] for row in ids)
    stats = native.ple_reader_stats(reader)
    block_bytes = stats["dio_align"]
    blocks = _touched_blocks(block_bytes, ((row * 4096, 4096) for row in set(ids)))
    assert stats["unique_blocks"] == len(blocks)
    assert stats["read_bytes"] == len(blocks) * block_bytes
    expected_calls, expected_coalesced = _coalesced_read_counts(blocks, block_bytes)
    assert stats["read_calls"] == expected_calls
    assert stats["coalesced_reads"] == expected_coalesced


def test_partial_final_block_and_truncation_drain_before_reuse(
    tmp_path, reader_factory
):
    native, create = reader_factory
    path = tmp_path / "partial"
    offset, row_bytes, rows = 107, 509, 80
    payload = bytes((i * 7) % 256 for i in range(rows * row_bytes))
    contents = b"h" * offset + payload
    path.write_bytes(contents)
    reader = create(rows, rows, 0, rows, row_bytes, 0, 8)
    native.ple_reader_add(reader, 0, str(path), offset, False)
    ids = array("q", [79, 0, 33, 51, 17, 79])
    output = bytearray(len(ids) * row_bytes)
    native.ple_reader_run(reader, ids, output, None, len(ids))
    expected = b"".join(payload[row * row_bytes : (row + 1) * row_bytes] for row in ids)
    assert output == expected
    with path.open("r+b") as stream:
        stream.truncate(len(contents) - 1)
    with pytest.raises(RuntimeError, match="short PLE read"):
        native.ple_reader_run(reader, ids, output, None, len(ids))
    # A failed transaction drains all CQEs: restore and reuse its slots.
    path.write_bytes(contents)
    native.ple_reader_run(reader, ids, output, None, len(ids))
    assert output == expected


def test_large_source_offsets_keep_exact_rows(tmp_path, reader_factory):
    native, create = reader_factory
    offset, rows, width = 2**32 + 4103, 8, 96
    payload = bytes((index * 17) % 256 for index in range(rows * width))
    path = tmp_path / "large_offset"
    with path.open("wb") as file:
        file.seek(offset)
        file.write(payload)
    reader = create(rows, rows, 0, rows, width, 0, 4)
    native.ple_reader_add(reader, 0, str(path), offset, False)
    ids = array("q", [7, 0, 7, 3])
    output = bytearray(len(ids) * width)
    native.ple_reader_run(reader, ids, output, None, len(ids))
    assert output == b"".join(payload[row * width : (row + 1) * width] for row in ids)


def test_sources_and_destinations_fail_before_unsafe_reads(tmp_path, reader_factory):
    native, create = reader_factory
    reader = create(4, 4, 0, 4, 16, 0, 4)
    path = tmp_path / "short"
    path.write_bytes(bytes(63))
    with pytest.raises(RuntimeError, match="range exceeds"):
        native.ple_reader_add(reader, 0, str(path), 0, False)
    ids = array("q", [0, 1, 2, 3])
    output = bytearray(64)
    with pytest.raises(RuntimeError, match="missing PLE weight source"):
        native.ple_reader_run(reader, ids, output, None, 4)
    path.write_bytes(bytes(range(64)))
    native.ple_reader_add(reader, 0, str(path), 0, False)
    with pytest.raises(RuntimeError, match="already registered"):
        native.ple_reader_add(reader, 0, str(path), 0, False)
    with pytest.raises(ValueError, match="capacity"):
        native.ple_reader_run(reader, ids, output, None, 5)
    with pytest.raises(ValueError, match="cover count rows"):
        native.ple_reader_run(reader, ids, output[:-1], None, 4)
    with pytest.raises((BufferError, TypeError)):
        native.ple_reader_run(reader, ids, bytes(64), None, 4)
    shared = bytearray(64)
    with pytest.raises(ValueError, match="must not overlap"):
        native.ple_reader_run(reader, memoryview(shared)[:32], shared, None, 4)
    native.ple_reader_run(reader, ids, output, None, 4)
    assert output == bytes(range(64))


def test_large_mixed_file_batches_preserve_order_and_reuse_allocations(
    tmp_path, reader_factory
):
    native, create = reader_factory
    shard_rows, shards = 2048, 2
    row_bytes, scale_bytes, count = 37, 11, 2080
    weight_payloads = {}
    scale_payloads = {}
    reader = create(
        shard_rows,
        shard_rows * shards,
        0,
        shard_rows * shards,
        row_bytes,
        scale_bytes,
        count,
    )
    for shard in range(shards):
        weights = bytes(
            (shard * 43 + index * 17 + index // 29) % 256
            for index in range(shard_rows * row_bytes)
        )
        scales = bytes(
            (shard * 71 + index * 13 + index // 7) % 256
            for index in range(shard_rows * scale_bytes)
        )
        weight_payloads[shard] = weights
        scale_payloads[shard] = scales
        weight_path = tmp_path / f"weights-{shard}"
        scale_path = tmp_path / f"scales-{shard}"
        weight_offset = 137 + shard * 19
        scale_offset = 281 + shard * 23
        weight_path.write_bytes(b"w" * weight_offset + weights)
        scale_path.write_bytes(b"s" * scale_offset + scales)
        native.ple_reader_add(reader, shard, str(weight_path), weight_offset, False)
        native.ple_reader_add(reader, shard, str(scale_path), scale_offset, True)

    ids = array(
        "q",
        ((index * 1543 + 97) % (shard_rows * shards) for index in range(count - 32)),
    )
    ids.extend(ids[:32])
    output = bytearray(count * row_bytes)
    output_scales = bytearray(count * scale_bytes)
    allocation = native.ple_reader_stats(reader)
    native.ple_reader_run(reader, ids, output, output_scales, count)
    assert output == _expected_rows(weight_payloads, ids, shard_rows, row_bytes)
    assert output_scales == _expected_rows(scale_payloads, ids, shard_rows, scale_bytes)
    after_large = native.ple_reader_stats(reader)
    assert after_large["staging_bytes"] == allocation["staging_bytes"]
    assert after_large["metadata_bytes"] == allocation["metadata_bytes"]

    short_ids = array("q", [4095, 0, 2048, 97, 4095, 2048, 1])
    native.ple_reader_run(reader, short_ids, output, output_scales, len(short_ids))
    assert output[: len(short_ids) * row_bytes] == _expected_rows(
        weight_payloads, short_ids, shard_rows, row_bytes
    )
    assert output_scales[: len(short_ids) * scale_bytes] == _expected_rows(
        scale_payloads, short_ids, shard_rows, scale_bytes
    )
    after_short = native.ple_reader_stats(reader)
    assert after_short["staging_bytes"] == allocation["staging_bytes"]
    assert after_short["metadata_bytes"] == allocation["metadata_bytes"]


def test_fragment_sort_is_skipped_only_when_no_merge_can_pay(tmp_path, reader_factory):
    """Density gate (miss_io.md E2).

    A wave is planned straight from fragment order only when no single file
    holds enough fragments for two of them to land on neighbouring sectors,
    because that is the only case where the sort can produce a merge. Rows are
    laid out as adjacent-sector pairs separated from their neighbours, so the
    sorted path provably merges every pair and the skipped path provably merges
    nothing, while the bytes read are identical either way.
    """
    native, create = reader_factory
    row_bytes, file_count, sector_count = 8, 32, 1024
    paths = [tmp_path / f"shard-{index}" for index in range(file_count)]

    def sparse(path, size):
        with path.open("wb") as stream:
            stream.truncate(size)

    def rows_of(pair_count, stride):
        # The two members of a pair are emitted far apart in wave order, so the
        # adjacency exists in the file but not in the fragment array: only a
        # sort can find it. Each pair starts two sectors after the previous
        # one's second sector, so pairs never chain into longer runs.
        low = [4 * pair * stride + 1 for pair in range(pair_count)]
        high = [(4 * pair + 1) * stride + 1 for pair in range(pair_count)]
        return low + high

    # Ask the reader which fragment floor it will use before sizing the files,
    # so the density arithmetic is exact for a 512 B and a 4 KiB floor alike.
    for path in paths:
        sparse(path, sector_count * 4096)
    probe_rows = sector_count * 4096 // row_bytes
    probe = create(
        probe_rows, probe_rows * file_count, 0, probe_rows * file_count, row_bytes, 0, 2
    )
    for index, path in enumerate(paths):
        native.ple_reader_add(probe, index, str(path), 0, False)
    gran = native.ple_reader_stats(probe)["dio_align"]
    probe = None

    file_bytes = sector_count * gran
    shard_rows = file_bytes // row_bytes
    stride = gran // row_bytes
    for path in paths:
        sparse(path, file_bytes)
    reader = create(
        shard_rows,
        shard_rows * file_count,
        0,
        shard_rows * file_count,
        row_bytes,
        0,
        file_count * 64,
    )
    for index, path in enumerate(paths):
        native.ple_reader_add(reader, index, str(path), 0, False)

    def run(pair_count):
        ids, expected = array("q"), bytearray()
        for index, path in enumerate(paths):
            with path.open("rb+") as stream:
                for row in rows_of(pair_count, stride):
                    stream.seek(row * row_bytes)
                    payload = bytes(
                        (17 * index + 91 * row + j) % 256 for j in range(row_bytes)
                    )
                    stream.write(payload)
                    expected += payload
                    ids.append(index * shard_rows + row)
        output = bytearray(b"\xff" * (len(ids) * row_bytes))
        native.ple_reader_run(reader, ids, output, None, len(ids))
        assert bytes(output) == bytes(expected)
        return native.ple_reader_stats(reader), len(ids)

    # 32 fragments per file over 1025 sectors: 32 * 31 < 1025, so no file
    # reaches even one expected merge and the sort is skipped although adjacency
    # is on offer. 1024 fragments total keeps the wave past PLE_SORT_MIN.
    stats, fragments = run(16)
    assert fragments == file_count * 32
    assert stats["read_calls"] == fragments
    assert stats["coalesced_reads"] == 0

    # 64 fragments per file: 64 * 63 >= 1025, so the sort runs and every pair
    # pays for itself with one fewer command.
    stats, fragments = run(32)
    assert fragments == file_count * 64
    assert stats["coalesced_reads"] == file_count * 32
    assert stats["read_calls"] == fragments - file_count * 32

    # The knobs that may silently fall back stay reported by stats.
    assert stats["depth_requested"] == 4
    assert 2 <= stats["depth_effective"] <= 256
    assert stats["dio_align"] in (512, 1024, 2048, 4096)
    assert stats["submit_calls"] > 0
