"""GPU byte and lifetime contracts for the selected immutable-file backend.

Set B12X_DISK_BACKEND=gds and place pytest's basetemp on a GDS filesystem to
qualify cuFile. A selected but unavailable transport fails instead of skipping.
"""

from __future__ import annotations

import os
import random
import threading
import time

import numpy as np
import pytest
import torch

from b12x.sequence._shared.disk_table import DiskRowCache
from tests.experimental.b12x.conftest import require_b12x


def _cache(**kwargs):
    queue_depth = kwargs.pop("queue_depth", 2)
    return DiskRowCache(
        device=require_b12x(),
        max_lookups=40,
        table_rows=300,
        shard_start=7,
        shard_end=290,
        shard_rows=100,
        weight_row_bytes=257,
        scale_row_bytes=9,
        queue_depth=queue_depth,
        **kwargs,
    )


def _write(path, payload, offset):
    with path.open("wb", buffering=0) as stream:
        stream.seek(offset)
        stream.write(payload)
        os.fsync(stream.fileno())


@torch.inference_mode()
@pytest.mark.parametrize("queue_depth", [2, 3])
def test_rows_match_bytes_across_planes_shards_streams_and_frozen_gather(
    tmp_path, monkeypatch, queue_depth
):
    cache = _cache(queue_depth=queue_depth)
    payloads = []
    try:
        for scale, width in [(False, 257), (True, 9)]:
            data = (
                (
                    torch.arange(300 * width, dtype=torch.int64) * 17
                    + torch.arange(300).repeat_interleave(width)
                )
                .remainder(251)
                .to(torch.uint8)
                .reshape(300, width)
            )
            payloads.append(data)
            for shard in range(3):
                path = tmp_path / f"{scale}-{shard}.bin"
                _write(
                    path, data[shard * 100 : (shard + 1) * 100].numpy().tobytes(), 4093
                )
                cache.add_shard(shard, str(path), 4093, scale=scale)
        cache.freeze()
        if cache._gds:
            from b12x.sequence._shared._gds import _gather

            monkeypatch.setattr(
                _gather,
                "run",
                lambda *a, **k: pytest.fail(
                    "live counts must reuse the compiled gather"
                ),
            )
            cache._gds.native.start_stats()
        pointers = [cache.weight.data_ptr(), cache.scale.data_ptr()]
        footprint = cache.stats()["owned_staging_bytes"]
        streams = [torch.cuda.Stream(device=cache.device) for _ in range(2)]
        ids = torch.tensor(
            [
                7,
                7,
                99,
                100,
                199,
                200,
                289,
                290,
                -1,
                6,
                299,
                300,
                2**40,
                15,
                16,
                31,
                32,
                47,
                48,
                79,
                80,
                127,
                128,
                191,
                192,
                255,
                256,
            ],
            device=cache.device,
        )
        out = torch.empty_like(cache.weight)
        for iteration, count in enumerate([27, 1, 0, 13, 27, 5]):
            stream = streams[iteration % 2]
            with torch.cuda.stream(stream), cache.transaction():
                cache.read_rows(ids, count)
                out[:count].copy_(cache.weight[:count])
                for actual, source in [(out, payloads[0]), (cache.scale, payloads[1])]:
                    host_ids = ids[:count].cpu()
                    expected = torch.zeros((count, source.shape[1]), dtype=torch.uint8)
                    valid = (host_ids >= 7) & (host_ids < 290)
                    expected[valid] = source[host_ids[valid]]
                    torch.testing.assert_close(
                        actual[:count].cpu(), expected, rtol=0, atol=0
                    )
            assert pointers == [cache.weight.data_ptr(), cache.scale.data_ptr()]
            assert cache.stats()["owned_staging_bytes"] == footprint
        if cache._gds:
            assert cache.weight_host is cache.scale_host is None
            stats = cache._gds.native.transport_stats()
            assert stats["nvfs_ops"] + stats["p2p_ops"] > 0
            assert all(
                stats[k] == 0
                for k in ("posix_ops", "aio_ops", "iouring_ops", "read_errors")
            )
    finally:
        cache.close()
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        with cache.transaction():
            pass


@torch.inference_mode()
def test_file_offsets_and_global_ids_exceed_signed_32bit_products(tmp_path):
    device = require_b12x()
    row = (1 << 32) // 256 + 3
    offset = 4093
    path = tmp_path / "large-offset.bin"
    data = torch.arange(256, dtype=torch.uint8).numpy().tobytes()
    _write(path, data, offset + row * 256)
    cache = DiskRowCache(
        device=device,
        max_lookups=2,
        table_rows=row + 1,
        shard_start=row,
        shard_end=row + 1,
        shard_rows=row + 1,
        weight_row_bytes=256,
        queue_depth=1,
    )
    try:
        cache.add_shard(0, str(path), offset)
        cache.freeze()
        with cache.transaction():
            cache.read_rows(torch.tensor([row, row - 1], device=device), 2)
            torch.testing.assert_close(
                cache.weight[0].cpu(),
                torch.arange(256, dtype=torch.uint8),
                rtol=0,
                atol=0,
            )
            assert cache.weight[1].count_nonzero().item() == 0
    finally:
        cache.close()


def test_source_and_read_failures_are_reported_before_reuse(tmp_path):
    device = require_b12x()
    path = tmp_path / "source.bin"
    _write(path, bytes(range(256)) * 64, 0)
    cache = DiskRowCache(
        device=device,
        max_lookups=2,
        table_rows=64,
        shard_start=0,
        shard_end=64,
        shard_rows=64,
        weight_row_bytes=256,
        queue_depth=2,
    )
    try:
        with pytest.raises(RuntimeError, match="open"):
            cache.add_shard(0, str(tmp_path / "missing"), 0)
        cache.add_shard(0, str(path), 0)
        cache.freeze()
        # Violating the immutable-file contract must report a short read.
        with path.open("r+b") as stream:
            stream.truncate(4096)
        with pytest.raises(RuntimeError, match="short|failed|unexpected"):
            with cache.transaction():
                cache.read_rows(torch.tensor([31, 63], device=device), 2)
        assert cache._transaction_thread is None
    finally:
        cache.close()


def test_backend_is_fixed_at_construction_and_optional_dependency_is_lazy(monkeypatch):
    from b12x.loader import _gds_native

    def unavailable():
        raise RuntimeError("injected unavailable cuFile development files")

    monkeypatch.setattr(_gds_native, "load", unavailable)
    monkeypatch.delenv("B12X_DISK_BACKEND", raising=False)
    cache = _cache()
    try:
        assert cache._backend == "io_uring"
        monkeypatch.setenv("B12X_DISK_BACKEND", "gds")
        assert cache._backend == "io_uring"
        with pytest.raises(RuntimeError, match="unavailable cuFile"):
            _cache()
        monkeypatch.setenv("B12X_DISK_BACKEND", "invalid")
        with pytest.raises(ValueError, match="B12X_DISK_BACKEND"):
            _cache()
    finally:
        cache.close()


def test_close_waits_for_a_consumer_on_another_stream(tmp_path):
    device = require_b12x()
    path = tmp_path / "pending.bin"
    _write(path, bytes(range(256)) * 16, 0)
    cache = DiskRowCache(
        device=device,
        max_lookups=1,
        table_rows=16,
        shard_start=0,
        shard_end=16,
        shard_rows=16,
        weight_row_bytes=256,
        queue_depth=1,
    )
    cache.add_shard(0, str(path), 0)
    cache.freeze()
    stream = torch.cuda.Stream(device=device)
    result = torch.empty((256,), dtype=torch.uint8, device=device)
    ids = torch.tensor([1], device=device)
    with torch.cuda.stream(stream), cache.transaction():
        cache.read_rows(ids, 1)
        torch.cuda._sleep(10_000_000)
        result.copy_(cache.weight[0])
    cache.close()
    assert stream.query()
    torch.testing.assert_close(
        result.cpu(), torch.arange(256, dtype=torch.uint8), rtol=0, atol=0
    )


def test_full_batch_uses_registered_slots_above_one_mib(tmp_path):
    device = require_b12x()
    path = tmp_path / "batch-slots.bin"
    data = ((torch.arange(4096)[:, None] * 17 + torch.arange(256)) % 251).to(
        torch.uint8
    )
    _write(path, data.numpy().tobytes(), 0)
    cache = DiskRowCache(
        device=device,
        max_lookups=64,
        table_rows=4096,
        shard_start=0,
        shard_end=4096,
        shard_rows=4096,
        weight_row_bytes=256,
        queue_depth=64,
    )
    try:
        cache.add_shard(0, str(path), 0)
        cache.freeze()
        for count in [64, 1, 63]:
            ids = torch.arange(count, device=device) * 32
            with cache.transaction():
                cache.read_rows(ids, count)
                torch.testing.assert_close(
                    cache.weight[:count].cpu(), data[ids.cpu()], rtol=0, atol=0
                )
            assert cache.stats()["read_calls"] == count
    finally:
        cache.close()


def _payloads():
    return [
        (
            torch.arange(300 * width, dtype=torch.int64) * 17
            + torch.arange(300).repeat_interleave(width)
        )
        .remainder(251)
        .to(torch.uint8)
        .reshape(300, width)
        for width in (257, 9)
    ]


def _expected(source, ids):
    expected = torch.zeros((len(ids), source.shape[1]), dtype=torch.uint8)
    valid = (ids >= 7) & (ids < 290)
    expected[valid] = source[ids[valid]]
    return expected


def _byte_identity_round_trip(tmp_path, monkeypatch, capacity, source):
    """Serve one batch pattern through the C S3-FIFO host cache and check
    every staged byte.

    Cached staging must equal the source for repeats, in-batch duplicates,
    invalid IDs, and capacities that evict rows the same batch still reads.
    The single shipped policy runs this identical fixture, so the cache
    cannot change what the read path returns.
    """
    monkeypatch.setenv("B12X_DISK_BACKEND", "io_uring")
    payloads = _payloads()
    host_rows = None
    if source == "host":
        host_rows = tuple(p[7:290].contiguous() for p in payloads)
    cache = _cache(cache_rows=capacity, cache_tier="host", host_rows=host_rows)
    try:
        if source == "disk":
            for scale, data in zip((False, True), payloads):
                for shard in range(3):
                    path = tmp_path / f"{scale}-{shard}.bin"
                    _write(
                        path, data[shard * 100 : (shard + 1) * 100].numpy().tobytes(), 0
                    )
                    cache.add_shard(shard, str(path), 0, scale=scale)
        cache.freeze()
        assert (cache.weight.device.type == "cuda") and cache.weight_host is not None
        generator = torch.Generator().manual_seed(capacity)
        hot = torch.tensor([7, 8, 150, 151, 289, 6, 290, 2**40])
        for count in [40, 40, 1, 0, 17, 40, 40]:
            ids = torch.where(
                torch.rand(count, generator=generator) < 0.6,
                hot[torch.randint(len(hot), (count,), generator=generator)],
                torch.randint(0, 300, (count,), generator=generator),
            )
            with cache.transaction():
                cache.read_rows(ids.to(cache.device), count)
                for actual, data in zip((cache.weight, cache.scale), payloads):
                    torch.testing.assert_close(
                        actual[:count].cpu(), _expected(data, ids), rtol=0, atol=0
                    )
        stats = cache.stats()
        assert stats["row_cache_tier"] == "host"
        assert 0 < stats["row_cache_resident"] <= capacity
        assert stats["row_cache_hits"] > 0
        # The deferred-free contract: the pool is only ever read/written
        # inside a transaction, whose _cache_done event fences the next one.
        assert cache._cache_used and cache._transaction_thread is None
    finally:
        cache.close()


@torch.inference_mode()
@pytest.mark.parametrize(
    ("capacity", "source"),
    [(c, s) for c in (1, 5, 64, 400) for s in ("disk", "host")],
)
def test_cached_rows_match_uncached_bytes(tmp_path, monkeypatch, capacity, source):
    """The C S3-FIFO host cache stages the source's bytes exactly.

    Eviction order is the reference class's, verified per request in
    ``test_row_cache_policies.py``; what this fixture pins is the byte
    contract: a resident row holds the source's bytes and a victim never
    clobbers a row the same batch reads.
    """
    _byte_identity_round_trip(tmp_path, monkeypatch, capacity, source)


@pytest.mark.parametrize("tier", ["unified", "ram"])
def test_invalid_row_cache_tier_fails_construction(monkeypatch, tier):
    """A non-``host`` tier is a construction error; there is one tier."""
    monkeypatch.delenv("B12X_ROW_CACHE_TIER", raising=False)
    host_rows = _host_rows(_payloads())
    with pytest.raises(ValueError, match="B12X_ROW_CACHE_TIER"):
        _cache(cache_rows=8, cache_tier=tier, host_rows=host_rows)
    monkeypatch.setenv("B12X_ROW_CACHE_TIER", tier)
    with pytest.raises(ValueError, match="B12X_ROW_CACHE_TIER"):
        _cache(cache_rows=8, host_rows=host_rows)


def test_s3fifo_c_index_selected_and_stats_report_host():
    """Construction wires the C S3-FIFO facade and stats carry its shape.

    ``stats()['row_cache_tier']`` is ``host`` when the cache is enabled; a
    disabled cache reports neither tier nor pool.
    """
    from b12x.sequence._shared.row_cache import S3FifoRowCache

    host_rows = _host_rows(_payloads())

    def build(**kwargs):
        return _cache(cache_rows=8, cache_tier="host", host_rows=host_rows, **kwargs)

    cache = build()
    try:
        assert type(cache._index) is S3FifoRowCache
        stats = cache.stats()
        assert stats["row_cache_tier"] == "host"
        assert stats["row_cache_rows"] == 8
        assert stats["row_cache_bytes"] == 8 * 266
    finally:
        cache.close()

    # A disabled cache reports no tier.
    uncached = _cache(cache_rows=0, cache_tier="host", host_rows=host_rows)
    try:
        assert uncached._index is None
        assert uncached.cache_tier is None
        assert "row_cache_tier" not in uncached.stats()
    finally:
        uncached.close()


def test_enabled_without_size_defaults_to_1gib(monkeypatch):
    """Any row-cache knob set non-empty with no size selects the 1 GiB pool,
    rounded down to whole rows (257 + 9 B/row here), on the C path.

    Explicit ``cache_rows`` args and BYTES/ROWS values are unaffected (the
    precedence tests for the pure helper live in
    ``test_row_cache_policies.py``).
    """
    from b12x.sequence._shared.row_cache import S3FifoRowCache

    for key in ("B12X_ROW_CACHE_ROWS", "B12X_ROW_CACHE_BYTES", "B12X_ROW_CACHE_TIER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("B12X_ROW_CACHE_TIER", "host")
    cache = _cache(host_rows=_host_rows(_payloads()))
    try:
        assert cache.cache_rows == (1 << 30) // 266
        assert type(cache._index) is S3FifoRowCache
        stats = cache.stats()
        assert stats["row_cache_rows"] == cache.cache_rows
        assert stats["row_cache_bytes"] == cache.cache_rows * 266
        assert stats["row_cache_tier"] == "host"
    finally:
        cache.close()

    # ROWS=0 is an explicit size and keeps the cache off.
    monkeypatch.setenv("B12X_ROW_CACHE_ROWS", "0")
    off = _cache(host_rows=_host_rows(_payloads()))
    try:
        assert off.cache_rows == 0
        assert off._index is None
    finally:
        off.close()


@pytest.mark.parametrize(
    ("text", "rows"),
    [
        # row_bytes=90: (8.6 * 2**20) // 90 == 100197, i.e. 8.6 MiB is the
        # capacity a 100k-row PLE pool (90 B/row) actually costs.
        ("8.6MiB", 100_197),
        ("8.6Mi", 100_197),
        ("8600000", 95_555),
        (" 1 GiB ", 11_930_464),
        ("1G", 11_930_464),
        ("1kib", 11),
        ("0.5KiB", 5),
        ("8600000B", 95_555),
    ],
)
def test_row_cache_size_parses_to_whole_rows(text, rows):
    from b12x.sequence._shared.disk_table import _parse_row_cache_size

    assert _parse_row_cache_size(text, 90) == rows


@pytest.mark.parametrize("text", ["", "MiB", "-1MiB", "1.2.3MiB", "0", "1PiB"])
def test_row_cache_size_rejects_unusable_values(text):
    """Malformed, sub-row and unmultipliable values are errors, not a silent off."""
    from b12x.sequence._shared.disk_table import _parse_row_cache_size

    with pytest.raises(ValueError, match="B12X_ROW_CACHE_BYTES"):
        _parse_row_cache_size(text, 90)


def test_row_cache_bytes_env_selects_capacity(monkeypatch):
    """The env knob sizes the pool in bytes; explicit rows and args win cleanly."""
    host_rows = _host_rows(_payloads())
    monkeypatch.delenv("B12X_ROW_CACHE_ROWS", raising=False)
    monkeypatch.delenv("B12X_ROW_CACHE_BYTES", raising=False)

    monkeypatch.setenv("B12X_ROW_CACHE_BYTES", "8.6MiB")
    cache = _cache(cache_tier="host", host_rows=host_rows)
    try:
        # 8.6 MiB at 257 + 9 B/row: rounds down to whole rows, never a partial slot.
        assert cache.cache_rows == (8.6 * 2**20) // 266
        assert cache.stats()["row_cache_rows"] == cache.cache_rows
    finally:
        cache.close()

    # The argument is an explicit operator choice and beats the environment.
    explicit = _cache(cache_rows=8, cache_tier="host", host_rows=host_rows)
    try:
        assert explicit.cache_rows == 8
    finally:
        explicit.close()

    monkeypatch.setenv("B12X_ROW_CACHE_ROWS", "16")
    with pytest.raises(ValueError, match="only one of"):
        _cache(cache_tier="host", host_rows=host_rows)

    # A request below one row is an error, not a silently disabled cache.
    monkeypatch.delenv("B12X_ROW_CACHE_ROWS")
    monkeypatch.setenv("B12X_ROW_CACHE_BYTES", "10")
    with pytest.raises(ValueError, match="rounds to zero rows"):
        _cache(cache_tier="host", host_rows=host_rows)


def test_cache_requires_io_uring_or_host_source(monkeypatch):
    monkeypatch.setenv("B12X_DISK_BACKEND", "gds")
    with pytest.raises(ValueError, match="GDS backend does not support a row cache"):
        _cache(cache_rows=8, cache_tier="host")


def _host_rows(payloads):
    """CPU uint8 host-resident shard tables over the cache's ``[7, 290)`` range."""
    return tuple(p[7:290].contiguous() for p in payloads)


def _fake_fetch(payloads, block=None, fail_first=0, record=None):
    """Deterministic stand-in for the source read, independent of disk/native.

    It writes ``payloads[plane][gid]`` for in-shard ids and zeros otherwise, so
    prefetched and synchronously fetched bytes are byte-identical for a given
    id (last write wins). ``block`` holds the worker inside the fetch so a test
    can observe loading slots; ``fail_first`` raises for the leading calls to
    exercise the cancel path; ``record`` collects the raw call arguments.
    """
    weight_src, scale_src = payloads[0].numpy(), payloads[1].numpy()
    state = {"calls": 0}

    def fetch(self, ids, weights, scales, count):
        state["calls"] += 1
        if record is not None:
            record.append((ids, weights, scales, count))
        if block is not None:
            block.wait(timeout=5)
        if state["calls"] <= fail_first:
            raise IOError("injected prefetch read failure")
        id_arr = np.frombuffer(ids, dtype=np.int64, count=count)
        for buffer, source in ((weights, weight_src), (scales, scale_src)):
            if buffer is None:
                continue
            width = source.shape[1]
            out = np.frombuffer(buffer, dtype=np.uint8, count=count * width).reshape(
                count, width
            )
            out.flags.writeable = True
            for row, gid in enumerate(id_arr):
                gid = int(gid)
                out[row] = source[gid] if 7 <= gid < 290 else 0

    return fetch


def _drain(cache, ids, timeout=3.0):
    index = cache._index
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cache._prefetch_queue.empty() and all(index.is_ready(gid) for gid in ids):
            return
        time.sleep(0.005)
    raise AssertionError(f"prefetch did not drain: {ids}")


def _wait_loading(cache, ids, timeout=3.0):
    index = cache._index
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        loading = index.debug_state()["loading"]
        if cache._prefetch_queue.empty() and all(gid in loading for gid in ids):
            return
    raise AssertionError(f"ids never entered loading: {ids}")


def _wait_jobs(cache, record, at_least, timeout=3.0):
    index = cache._index
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (
            len(record) >= at_least
            and cache._prefetch_queue.empty()
            and not index.debug_state()["loading"]
        ):
            return
        time.sleep(0.005)
    raise AssertionError(f"prefetch jobs did not settle (calls={len(record)})")


def test_prefetch_disabled_returns_false(monkeypatch):
    """Env unset / "0" and a zero-row cache never fetch and never spawn a thread."""
    monkeypatch.delenv("B12X_PLE_PREFETCH", raising=False)
    payloads = _payloads()
    host_rows = _host_rows(payloads)
    before = threading.active_count()
    cache = _cache(cache_rows=16, cache_tier="host", host_rows=host_rows)
    try:
        assert cache.prefetch([7, 8, 9]) is False
        assert cache._prefetch_thread is None
        monkeypatch.setenv("B12X_PLE_PREFETCH", "0")
        assert cache.prefetch([7, 8, 9]) is False
        assert cache._prefetch_thread is None
    finally:
        cache.close()
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    zero = _cache(cache_rows=0, cache_tier="host", host_rows=host_rows)
    try:
        assert zero.prefetch([7, 8]) is False
        assert zero._prefetch_thread is None
    finally:
        zero.close()
    assert threading.active_count() == before


@torch.inference_mode()
def test_prefetch_ready_hits_after_drain(monkeypatch):
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    monkeypatch.setattr(DiskRowCache, "_fetch", _fake_fetch(payloads))
    cache = _cache(cache_rows=32, cache_tier="host", host_rows=_host_rows(payloads))
    try:
        cache.freeze()
        ids = [7, 8, 100, 150, 200, 289]
        assert cache.prefetch(ids) is True
        assert cache._prefetch_thread is not None
        _drain(cache, ids)
        assert all(cache._index.is_ready(gid) for gid in ids)
        with cache._lock:
            plan = cache._index.plan(list(ids))
        assert plan.miss_ids == []
        assert sorted(plan.hit_pos) == list(range(len(ids)))
        host_ids = torch.tensor(ids, dtype=torch.int64)
        with cache.transaction():
            cache.read_rows(host_ids.to(cache.device), len(ids))
            for actual, source in zip((cache.weight, cache.scale), payloads):
                torch.testing.assert_close(
                    actual[: len(ids)].cpu(),
                    _expected(source, host_ids),
                    rtol=0,
                    atol=0,
                )
    finally:
        cache.close()


def _assert_loading_slot_immune(monkeypatch, cache_rows):
    """A row a prefetch is still fetching is invisible and never a victim.

    ``cache_rows`` is sized for S3-FIFO: it keeps only its small-queue budget
    of rows that have not earned a second read yet, so 32 rows (S = 3) is the
    smallest cache that can hold all three prefetched ids once the
    synchronous batch has run through S.
    """
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    gate = threading.Event()
    monkeypatch.setattr(DiskRowCache, "_fetch", _fake_fetch(payloads, block=gate))
    cache = _cache(
        cache_rows=cache_rows, cache_tier="host", host_rows=_host_rows(payloads)
    )
    try:
        cache.freeze()
        loading_ids = [7, 8, 9]
        assert cache.prefetch(loading_ids) is True
        _wait_loading(cache, loading_ids)
        slots = cache._index.debug_state()["loading"]
        assert set(slots) == set(loading_ids)
        # Synchronous misses must evict committed rows, never a loading one.
        with cache._lock:
            plan = cache._index.plan([100, 101, 102, 103, 104, 105, 106, 107, 108])
        assert len(plan.miss_ids) == 9
        assert len(cache._index.debug_state()["ready"]) > 0
        # A store slot is a row this batch owns; a loading slot is in neither
        # queue, so the two can never name the same pool row.
        assert set(plan.store_slot).isdisjoint(slots.values())
        for gid in loading_ids:
            assert gid in cache._index.debug_state()["loading"]
            assert not cache._index.is_ready(gid)
            assert cache._index.debug_state()["loading"][gid] == slots[gid]
        gate.set()
        _drain(cache, loading_ids)
        assert all(cache._index.is_ready(gid) for gid in loading_ids)
    finally:
        gate.set()
        cache.close()


@torch.inference_mode()
def test_s3fifo_loading_slot_immune_to_eviction(monkeypatch):
    _assert_loading_slot_immune(monkeypatch, 32)


def test_duplicate_invalid_noop(monkeypatch):
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    record = []
    monkeypatch.setattr(DiskRowCache, "_fetch", _fake_fetch(payloads, record=record))
    cache = _cache(cache_rows=16, cache_tier="host", host_rows=_host_rows(payloads))
    try:
        cache.freeze()
        request = np.array([7, 7, 8, 8, 6, 290, 300, -5, 2**40, 100], dtype=np.int64)
        assert cache.prefetch(request) is True
        _drain(cache, [7, 8, 100])
        assert cache._index.is_ready(7) and cache._index.is_ready(8)
        assert cache._index.is_ready(100)
        for bad in (6, 290, 300, -5, 2**40):
            assert not cache._index.is_ready(bad)
        # One job, exactly the distinct in-shard ids, fetched once each.
        assert len(record) == 1
        fetched = set(
            np.frombuffer(record[0][0], dtype=np.int64, count=record[0][3]).tolist()
        )
        assert fetched == {7, 8, 100}
        # Re-prefetching already-ready rows is a no-op: still a single fetch.
        assert cache.prefetch([7, 8, 100]) is True
        assert len(record) == 1
    finally:
        cache.close()


def test_close_stops_worker(monkeypatch):
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    monkeypatch.setattr(DiskRowCache, "_fetch", _fake_fetch(payloads))
    cache = _cache(cache_rows=16, cache_tier="host", host_rows=_host_rows(payloads))
    cache.freeze()
    assert cache.prefetch([7, 8, 9]) is True
    worker = cache._prefetch_thread
    assert worker is not None
    started = time.monotonic()
    cache.close()
    elapsed = time.monotonic() - started
    assert not worker.is_alive()
    assert elapsed <= 2.5


@torch.inference_mode()
def test_rejected_prefetch_sync_identical(monkeypatch):
    monkeypatch.setenv("B12X_PLE_PREFETCH", "1")
    payloads = _payloads()
    record = []
    # The worker's first fetch fails; the synchronous read must still succeed.
    monkeypatch.setattr(
        DiskRowCache, "_fetch", _fake_fetch(payloads, fail_first=1, record=record)
    )
    cache = _cache(cache_rows=16, cache_tier="host", host_rows=_host_rows(payloads))
    try:
        cache.freeze()
        ids = [7, 8, 100]
        assert cache.prefetch(ids) is True
        _wait_jobs(cache, record, 1)
        # A rejected prefetch leaves nothing cached and nothing in flight.
        for gid in ids:
            assert not cache._index.is_ready(gid)
            assert gid not in cache._index.debug_state()["loading"]
        host_ids = torch.tensor(ids, dtype=torch.int64)
        with cache.transaction():
            cache.read_rows(host_ids.to(cache.device), len(ids))
            for actual, source in zip((cache.weight, cache.scale), payloads):
                torch.testing.assert_close(
                    actual[: len(ids)].cpu(),
                    _expected(source, host_ids),
                    rtol=0,
                    atol=0,
                )
    finally:
        cache.close()
