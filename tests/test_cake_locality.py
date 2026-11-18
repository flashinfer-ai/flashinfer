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

from __future__ import annotations

import gc

import pytest
import torch

from flashinfer import cake_locality as cl


# --------------------------------------------------------------------------- rules


@pytest.mark.parametrize(
    "units, counts, expected",
    [
        (7, (100, 112), (3, 4)),
        (7, (106, 106), (4, 3)),
        (1, (110, 102), (1, 0)),
        (1, (102, 110), (0, 1)),
        (0, (104, 108), (0, 0)),
        (64, (104, 108), (31, 33)),
    ],
)
def test_split_units(units, counts, expected):
    assert cl.split_units(units, counts) == expected


@pytest.mark.parametrize("counts", [(100, 112), (106, 106), (110, 102), (0, 212)])
def test_split_units_shares_every_unit(counts):
    for units in range(0, 130):
        u0, u1 = cl.split_units(units, counts)
        assert u0 + u1 == units and u0 >= 0 and u1 >= 0
        exact0 = units * counts[0] / sum(counts)
        assert abs(u0 - exact0) < 1


@pytest.mark.parametrize("units, counts", [(-1, (1, 1)), (1, (0, 0)), (1, (1,))])
def test_split_units_rejects(units, counts):
    with pytest.raises(ValueError):
        cl.split_units(units, counts)


def test_die_class():
    assert cl.die_class((106, 106)) == "even"
    assert cl.die_class((104, 108)) == "even"
    assert cl.die_class((108, 104)) == "even"
    assert cl.die_class((105, 109)) == "even"  # a gap of 4 SMs is even
    assert cl.die_class((104, 109)) == "uneven"
    assert cl.die_class((102, 110)) == "uneven"
    assert cl.die_class((112, 100)) == "uneven"


def _decode(word: int) -> tuple[int, int, int, int]:
    mask = (1 << cl.ENTRY_BITS) - 1
    return (
        word >> cl.ENTRY_RANK_SHIFT & mask,
        word >> cl.ENTRY_COUNT_SHIFT & mask,
        word >> cl.ENTRY_DOMAIN_SHIFT & ((1 << cl.ENTRY_DOMAIN_BITS) - 1),
        word >> cl.ENTRY_SUPPLEMENTAL_SHIFT & 1,
    )


def _check_words(counts, words, sm_count):
    assert len(words) == cl.SM_TABLE_LENGTH
    assert sum(counts) == sm_count
    seen = {0: [], 1: []}
    for word in words:
        if word == 0:
            continue
        rank, count, domain, _ = _decode(int(word))
        assert domain in (0, 1)
        assert count == counts[domain]
        seen[domain].append(rank)
    for domain in (0, 1):
        assert sorted(seen[domain]) == list(range(counts[domain]))


def test_pack_rows():
    rows = [(sm, sm % 2, int(sm == 5), 10) for sm in range(10)]
    counts, words = cl.pack_rows(rows)
    assert counts == (5, 5)
    _check_words(counts, words, 10)
    assert _decode(words[5]) == (2, 5, 1, 1)
    assert cl.pack_rows(rows + [(3, 1, 0, 10)]) is None  # an SM id twice
    assert cl.pack_rows([(sm, 0, 0, 4) for sm in range(4)]) is None  # one die
    assert cl.pack_rows([(256, 0, 0, 4), (1, 1, 0, 4)]) is None  # SM id too large
    assert cl.pack_rows([(0, 0, 0, 257), (1, 1, 0, 257)]) is None  # nsmid too large


class _Probes:
    """A stand-in probe returning the given ``(table, cacheable)`` outcomes."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, device_index):
        self.calls += 1
        return self.outcomes.pop(0)


def _fake_device(monkeypatch, probe):
    monkeypatch.setattr(cl, "_is_locality_arch", lambda index: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(cl, "_probe", lambda device_index: probe(device_index))


@pytest.mark.parametrize("cached", [False, True])
def test_sm_table_caches_only_repeatable_outcomes(monkeypatch, cached):
    """A probe miss or a probe that raised is tried again by the next call; a
    table, or an outcome a later probe would repeat (an old driver, invalid rows),
    is kept."""
    table = cl.SmTable(device_index=5, counts=(106, 106), entries=None)
    probe = _Probes((None, cached), (table, True))
    _fake_device(monkeypatch, probe)
    try:
        assert cl.sm_table(5) is None
        assert (5 in cl._TABLES) == cached
        second = cl.sm_table(5)
        assert second is (None if cached else table)
        assert probe.calls == (1 if cached else 2)
        assert cl.sm_table(5) is second
        assert probe.calls == (1 if cached else 2)
    finally:
        cl._forget()


def test_sm_table_one_probe_per_device(monkeypatch):
    """Concurrent first calls on one device run one probe."""
    import threading
    import time

    table = cl.SmTable(device_index=6, counts=(106, 106), entries=None)
    entered = []

    def slow_probe(device_index):
        entered.append(device_index)
        time.sleep(0.2)
        return table, True

    _fake_device(monkeypatch, slow_probe)
    results = []
    try:
        threads = [
            threading.Thread(target=lambda: results.append(cl.sm_table(6)))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert entered == [6]
        assert results == [table] * 4
    finally:
        cl._forget()


def test_probe_module_failure_is_not_cached(monkeypatch):
    """A probe library that fails to build or load is tried again; a toolkit
    older than LOCALITY_CUDA is remembered."""

    def broken():
        raise RuntimeError("build failed")

    try:
        cl._forget()
        monkeypatch.setattr(cl, "_toolkit_supported", lambda: True)
        monkeypatch.setattr(cl, "_module", broken)
        assert cl._probe_module() == (None, False)
        assert not cl._PROBE_MODULE
        monkeypatch.setattr(cl, "_toolkit_supported", lambda: False)
        assert cl._probe_module() == (None, True)
        assert cl._PROBE_MODULE == [None]
    finally:
        cl._forget()


def test_force_table():
    try:
        table = cl._force_table((102, 110), device_index=7)
        assert table.counts == (102, 110)
        assert table.entries.dtype == torch.int32
        assert table.entries.device.type == "cpu"
        _check_words(table.counts, table.entries.tolist(), 212)
    finally:
        cl._forget()


# --------------------------------------------------------------------------- device


def _require_locality() -> int:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device) != (10, 7):
        pytest.skip("Cake locality domains need sm_107a")
    from flashinfer.jit.cpp_ext import get_cuda_version

    version = get_cuda_version()
    if (version.major, version.minor) < cl.LOCALITY_CUDA:
        pytest.skip(f"Cake locality domains need a CUDA 13.5 toolkit, found {version}")
    from flashinfer.locality_domain import locality_domain_unsupported_reason

    reason = locality_domain_unsupported_reason(device)
    if reason is not None:
        pytest.skip(f"CUDA locality domains are unavailable: {reason}")
    # From here on, a build or probe failure is a test failure, not a skip.
    driver = int(cl._module().driver_version())
    if (driver // 1000, driver % 1000 // 10) < cl.LOCALITY_CUDA:
        pytest.skip(f"Cake locality domains need a CUDA 13.5 driver, found {driver}")
    return device


def test_sm_table_probe():
    device = _require_locality()
    cl._forget()
    table = cl.sm_table(device)
    assert table is not None
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    assert table.device_index == device
    assert min(table.counts) > 0
    assert table.entries.shape == (cl.SM_TABLE_LENGTH,)
    assert table.entries.dtype == torch.int32
    assert table.entries.device.type == "cpu"
    assert table.entries.is_contiguous()
    words = table.entries.tolist()
    _check_words(table.counts, words, sm_count)
    assert sum(word != 0 for word in words) == sm_count
    assert cl.sm_table(device) is table  # cached
    assert cl.sm_table(torch.device("cuda", device)) is table


def test_sm_table_not_probed_under_graph_capture():
    device = _require_locality()
    cl._forget()
    x = torch.zeros(16, device=f"cuda:{device}")
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        x.add_(1)
        captured = cl.sm_table(device)
    assert captured is None
    assert device not in cl._TABLES  # the capture outcome is not cached
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(x, torch.ones_like(x))
    table = cl.sm_table(device)
    assert table is not None
    # A cached table is returned under capture too.
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph2, stream=stream):
        x.add_(1)
        assert cl.sm_table(device) is table


def _domains(device):
    from flashinfer.locality_domain import get_localized_domains

    return get_localized_domains(device)


def _check_homing(tensor, cut, device):
    """First and last byte of each nonempty side are on the expected die, by both
    the Cake query and FlashInfer's own LocalizedDomains.pointer_domain."""
    base = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    flat = tensor.view(-1).view(torch.uint8)
    domains = _domains(device)
    sides = [(0, cut, 0), (cut, nbytes, 1)]
    for start, end, die in sides:
        if start == end:
            continue
        for offset in (start, end - 1):
            assert cl.pointer_domain(base + offset, device) == die, (offset, cut)
            assert domains.pointer_domain(flat[offset:]) == die, (offset, cut)


@pytest.mark.parametrize("parts", [(1, 3), (2, 2), (3, 1)])
def test_split_empty_homes_each_side(parts):
    device = _require_locality()
    unit = cl.granularity(device)
    assert unit > 0
    cut = parts[0] * unit
    nbytes = (parts[0] + parts[1]) * unit - 4096  # a tail shorter than the granule
    tensor = cl.split_empty((nbytes // 4,), torch.float32, cut, device)
    assert tensor.shape == (nbytes // 4,)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device("cuda", device)
    assert tensor.is_contiguous()
    _check_homing(tensor, cut, device)
    home = cl.homing(tensor)
    assert home == cl.Homing(
        device_index=device,
        base=tensor.data_ptr(),
        nbytes=nbytes,
        size=(parts[0] + parts[1]) * unit,
        cut=cut,
    )
    assert cl.homing(tensor[5:]) == home  # a view resolves through its storage
    # The returned tensor carries its record; its views and copies do not.
    assert cl.tensor_homing(tensor) == home
    assert cl.tensor_homing(tensor[5:]) is None
    assert cl.tensor_homing(tensor[:]) is None
    assert cl.tensor_homing(tensor.clone()) is None
    assert cl.homing(tensor.data_ptr()) == home
    assert cl.homing(tensor.data_ptr() + 4) is None
    # The whole range reads back what was written across the cut.
    values = torch.arange(tensor.numel(), dtype=torch.float32, device=tensor.device)
    tensor.copy_(values)
    torch.cuda.synchronize(device)
    assert torch.equal(tensor, values)


def test_split_empty_shaped_dtype():
    device = _require_locality()
    unit = cl.granularity(device)
    groups, n, k = 4, 1024, unit // 1024  # one granule per group
    tensor = cl.split_empty((groups, n, k), torch.float8_e4m3fn, 3 * unit, device)
    assert tensor.shape == (groups, n, k)
    assert tensor.dtype == torch.float8_e4m3fn
    _check_homing(tensor, 3 * unit, device)
    assert _domains(device).pointer_domain(tensor[2]) == 0
    assert _domains(device).pointer_domain(tensor[3]) == 1


@pytest.mark.parametrize("whole_die", [0, 1])
def test_split_empty_whole_on_one_die(whole_die):
    device = _require_locality()
    unit = cl.granularity(device)
    nbytes = 2 * unit + 512
    cut = nbytes if whole_die == 0 else 0
    tensor = cl.split_empty((nbytes,), torch.uint8, cut, device)
    _check_homing(tensor, cut, device)
    home = cl.homing(tensor)
    assert home is not None and home.cut == cut and home.size == 3 * unit
    tensor.fill_(7)
    torch.cuda.synchronize(device)
    assert int(tensor.sum()) == 7 * nbytes


def test_split_empty_rejects_invalid():
    device = _require_locality()
    unit = cl.granularity(device)
    with pytest.raises(ValueError):
        cl.split_empty((4 * unit,), torch.uint8, unit // 2, device)
    with pytest.raises(ValueError):
        cl.split_empty((4 * unit,), torch.uint8, 5 * unit, device)
    with pytest.raises(ValueError):
        cl.split_empty((0,), torch.uint8, 0, device)


def test_split_empty_free_and_registry_cleanup():
    device = _require_locality()
    unit = cl.granularity(device)
    # Exclude one-time CUDA/PyTorch initialization from the baseline.
    warm = cl.split_empty((unit,), torch.uint8, unit, device)
    warm.zero_()
    del warm
    gc.collect()
    torch.cuda.synchronize(device)
    free_before, _ = torch.cuda.mem_get_info(device)
    nbytes = 256 * 1024**2
    tensor = cl.split_empty((nbytes,), torch.uint8, nbytes // 2 // unit * unit, device)
    tensor.zero_()
    base = tensor.data_ptr()
    view = tensor[unit:]
    assert cl.homing(base) is not None
    del tensor
    gc.collect()
    assert cl.homing(base) is not None  # the view keeps it alive
    assert cl.homing(view) is not None
    torch.cuda.synchronize(device)
    del view
    gc.collect()
    assert cl.homing(base) is None
    free_after, _ = torch.cuda.mem_get_info(device)
    assert free_after >= free_before - 16 * 1024**2


def test_split_empty_cuda_graph_replay():
    device = _require_locality()
    unit = cl.granularity(device)
    source = cl.split_empty((2 * unit // 4,), torch.float32, unit, device)
    source.copy_(
        torch.arange(source.numel(), device=source.device, dtype=torch.float32)
    )
    out = torch.empty_like(source)
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        torch.mul(source, 2, out=out)
    source.add_(1)
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(out, source * 2, rtol=0, atol=0)


def test_counters():
    device = _require_locality()
    c0, c1 = cl.counters(2, device)
    for die, tensor in enumerate((c0, c1)):
        assert tensor.shape == (2,)
        assert tensor.dtype == torch.int32
        assert tensor.device == torch.device("cuda", device)
        assert int(tensor.abs().sum()) == 0
        assert cl.pointer_domain(tensor.data_ptr(), device) == die
    assert c0.data_ptr() != c1.data_ptr()


# --------------------------------------------------------------------------- launch counters


def _require_cuda() -> int:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.cuda.current_device()


def _allocations(device):
    return torch.cuda.memory_stats(device).get("allocation.all.allocated", 0)


@pytest.fixture
def fresh_counters():
    cl._forget_counters()
    yield
    cl._forget_counters()


def _unhomed(monkeypatch):
    """Counters without FlashInfer's locality-domain pools (another allocator
    backend, no MemPool, no locality domains): ordinary zeroed tensors."""
    from flashinfer import locality_domain

    def refuse(_index):
        raise locality_domain.LocalityDomainError("backend:cudaMallocAsync")

    monkeypatch.setattr(locality_domain, "_domain_pools", refuse)


@pytest.mark.parametrize("homed", [True, False], ids=["homed", "unhomed"])
def test_launch_counters_reused_per_stream(fresh_counters, monkeypatch, homed):
    """One stream reuses its pair without allocating; two streams get two pairs;
    a pair is zero, 16-byte aligned and, when homed, on each die."""
    device = _require_locality() if homed else _require_cuda()
    if not homed:
        _unhomed(monkeypatch)
    first = cl.launch_counters(2, device)
    torch.cuda.synchronize(device)
    before = _allocations(device)
    again = cl.launch_counters(2, device)
    assert _allocations(device) == before
    assert again[0] is first[0] and again[1] is first[1]
    side = torch.cuda.Stream(device=device)
    with torch.cuda.stream(side):
        other = cl.launch_counters(2, device)
        assert cl.launch_counters(2, device)[0] is other[0]
    pointers = {int(t.data_ptr()) for t in (*first, *other)}
    assert len(pointers) == 4
    torch.cuda.synchronize(device)
    for pair in (first, other):
        for die, tensor in enumerate(pair):
            assert tensor.shape == (2,) and tensor.dtype == torch.int32
            assert int(tensor.data_ptr()) % 16 == 0
            assert int(tensor.abs().sum()) == 0
            if homed:
                assert cl.pointer_domain(tensor.data_ptr(), device) == die


def test_launch_counters_cache_is_bounded(fresh_counters, monkeypatch):
    """Past STREAM_PAIR_LIMIT the least recently used stream's pair is dropped."""
    device = _require_cuda()
    _unhomed(monkeypatch)
    monkeypatch.setattr(cl, "STREAM_PAIR_LIMIT", 3)
    streams = [torch.cuda.Stream(device=device) for _ in range(5)]
    handles = {s.cuda_stream for s in streams}
    keep = cl.launch_counters(2, device)
    first = streams[0]
    with torch.cuda.stream(first):
        kept = cl.launch_counters(2, device)
    for stream in streams[1:]:
        with torch.cuda.stream(stream):
            cl.launch_counters(2, device)
        with torch.cuda.stream(first):
            assert cl.launch_counters(2, device)[0] is kept[0]  # recently used
        assert cl.launch_counters(2, device)[0] is keep[0]  # per thread: kept
    pairs = cl._COUNTER_STATES[device].stream_pairs
    assert len(pairs) == 3
    assert {key[0] for key in pairs} == {
        first.cuda_stream,
        streams[-2].cuda_stream,
        streams[-1].cuda_stream,
    }
    assert {key[0] for key in pairs} <= handles


def test_launch_counters_default_stream_pairs_are_per_thread(
    fresh_counters, monkeypatch
):
    """The default stream's handle may name a different stream in each thread:
    each thread keeps its own pair for it, reused by that thread."""
    import threading

    device = _require_cuda()
    _unhomed(monkeypatch)
    assert torch.cuda.current_stream(device).cuda_stream == 0
    mine = cl.launch_counters(2, device)
    assert cl.launch_counters(2, device)[0] is mine[0]
    seen = []

    def call():
        torch.cuda.set_device(device)
        pair = cl.launch_counters(2, device)
        assert cl.launch_counters(2, device)[0] is pair[0]
        seen.append(int(pair[0].data_ptr()))
        torch.cuda.synchronize(device)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
        thread.join()
    assert len(seen) == 2 and int(mine[0].data_ptr()) not in seen
    assert not cl._COUNTER_STATES[device].stream_pairs
    assert {0, 1, 2} == cl._THREAD_KEYED_HANDLES


def test_launch_counters_out_of_memory_pair_not_cached(fresh_counters, monkeypatch):
    """A pair made unhomed because a locality-domain pool was out of memory serves
    its call and is not cached; the next call tries the pools again."""
    from flashinfer import locality_domain

    device = _require_cuda()
    calls = []

    def exhausted(_index):
        calls.append(_index)
        raise torch.OutOfMemoryError("pool exhausted")

    monkeypatch.setattr(locality_domain, "_domain_pools", exhausted)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        first = cl.launch_counters(2, device)
        second = cl.launch_counters(2, device)
    torch.cuda.synchronize(device)
    assert first[0] is not second[0]
    assert all(int(t.abs().sum()) == 0 for t in (*first, *second))
    assert not cl._COUNTER_STATES[device].stream_pairs
    assert len(calls) >= 2 and device not in cl._POOLS_UNAVAILABLE
    _unhomed(monkeypatch)  # a LocalityDomainError is remembered and cached
    with torch.cuda.stream(stream):
        third = cl.launch_counters(2, device)
        assert cl.launch_counters(2, device)[0] is third[0]
    assert device in cl._POOLS_UNAVAILABLE


def test_launch_counters_refill_failure_does_not_fail_the_call(
    fresh_counters, monkeypatch
):
    """A capture reserve refill that raises leaves the call's pair intact."""
    device = _require_cuda()
    _unhomed(monkeypatch)

    def broken(_words, _index):
        raise RuntimeError("refill failed")

    monkeypatch.setattr(cl, "_settled_pair", broken)
    pair = cl.launch_counters(2, device)
    torch.cuda.synchronize(device)
    assert all(int(t.abs().sum()) == 0 for t in pair)
    reserve = cl._COUNTER_STATES[device].reserves[2]
    assert not reserve.refilling and not reserve.free
    monkeypatch.undo()
    _unhomed(monkeypatch)
    assert cl.launch_counters(2, device)[0] is pair[0]
    assert len(reserve.free) == cl.CAPTURE_RESERVE


def test_launch_counters_refill_waits_only_for_its_own_stream(
    fresh_counters, monkeypatch
):
    """The reserve is zeroed on the device's private stream: a refill waits for
    that stream, never for the caller's stream, and holds no lock while waiting."""
    device = _require_cuda()
    _unhomed(monkeypatch)
    waited = []
    synchronize = torch.cuda.Stream.synchronize

    def record(stream):
        state = cl._COUNTER_STATES[device]
        waited.append((int(stream.cuda_stream), state.lock.locked()))
        return synchronize(stream)

    monkeypatch.setattr(torch.cuda.Stream, "synchronize", record)
    caller = torch.cuda.Stream(device=device)
    with torch.cuda.stream(caller):
        cl.reserve_launch_counters(2, device)
        cl.launch_counters(2, device)
    side = cl._COUNTER_STATES[device].side_stream
    assert waited and all(handle == side.cuda_stream for handle, _ in waited)
    assert not any(locked for _, locked in waited)
    assert side.cuda_stream not in (caller.cuda_stream, 0)


def test_launch_counters_threads_share_one_streams_pair(fresh_counters, monkeypatch):
    """Threads launching on one stream share its pair; the cache stays consistent."""
    import threading

    device = _require_cuda()
    _unhomed(monkeypatch)
    stream = torch.cuda.Stream(device=device)
    seen = []
    barrier = threading.Barrier(8)

    def call():
        torch.cuda.set_device(device)
        with torch.cuda.stream(stream):
            barrier.wait()
            seen.append(int(cl.launch_counters(2, device)[0].data_ptr()))

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(seen) == 8 and len(set(seen)) == 1


@pytest.mark.parametrize("homed", [True, False], ids=["homed", "unhomed"])
def test_launch_counters_capture_takes_a_reserved_pair(
    fresh_counters, monkeypatch, homed
):
    """A captured call takes a pre-zeroed reserved pair (no allocation, so no
    memset node) that no eager stream and no other capture uses; the first call
    of a process, if captured, zeroes its pair inside the capture."""
    device = _require_locality() if homed else _require_cuda()
    if not homed:
        _unhomed(monkeypatch)
    stream = torch.cuda.Stream(device=device)
    # No eager call yet: no reserve, the capture zeroes its own pair.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        early = cl.launch_counters(2, device)
        assert cl._capture_id(device) != 0
    assert cl._capture_id(device) == 0
    graph.replay()
    torch.cuda.synchronize(device)
    assert all(int(t.abs().sum()) == 0 for t in early)
    eager = cl.launch_counters(2, device)  # fills the reserve
    with torch.cuda.stream(stream):
        eager_side = cl.launch_counters(2, device)
    reserve = cl._COUNTER_STATES[device].reserves[2]
    assert len(reserve.free) >= cl.CAPTURE_RESERVE
    captured = []
    for _ in range(2):
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            before = _allocations(device)
            captured.append(cl.launch_counters(2, device))
            assert _allocations(device) == before
    assert reserve.issued == 2
    pointers = [
        int(t.data_ptr()) for pair in (eager, eager_side, *captured) for t in pair
    ]
    assert len(set(pointers)) == len(pointers)
    for pair in captured:
        for die, tensor in enumerate(pair):
            assert int(tensor.data_ptr()) % 128 == 0
            assert int(tensor.abs().sum()) == 0
            if homed:
                assert cl.pointer_domain(tensor.data_ptr(), device) == die
    # The eager pair of the capture stream is still that stream's pair.
    with torch.cuda.stream(stream):
        assert cl.launch_counters(2, device)[0] is eager_side[0]


def test_launch_counters_reserve_refills_and_is_bounded(fresh_counters, monkeypatch):
    """Captures drain the reserve; the next eager call refills it once half is
    gone; past CAPTURE_PAIR_LIMIT captures zero their own pair."""
    device = _require_cuda()
    _unhomed(monkeypatch)
    monkeypatch.setattr(cl, "CAPTURE_RESERVE", 4)
    monkeypatch.setattr(cl, "CAPTURE_PAIR_LIMIT", 6)
    cl.launch_counters(2, device)
    reserve = cl._COUNTER_STATES[device].reserves[2]
    assert (len(reserve.free), reserve.made) == (4, 4)
    # Pairs of one block are 128 bytes apart: no two share an L2 line.
    starts = sorted(int(pair[0].data_ptr()) for pair in reserve.free)
    assert all(b - a == 128 for a, b in zip(starts, starts[1:], strict=False))
    stream = torch.cuda.Stream(device=device)
    graphs = []

    def capture():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            pair = cl.launch_counters(2, device)
        graphs.append(graph)
        return pair

    capture()
    cl.launch_counters(2, device)  # 3 of 4 left: no refill
    assert (len(reserve.free), reserve.made) == (3, 4)
    capture()
    cl.launch_counters(2, device)  # 2 of 4 left: refill to 4
    assert (len(reserve.free), reserve.made) == (4, 6)
    for _ in range(4):
        capture()
    cl.launch_counters(2, device)  # the limit: nothing more is made
    assert (len(reserve.free), reserve.made, reserve.issued) == (0, 6, 6)
    beyond = capture()  # zeroed inside its capture
    assert reserve.issued == 6
    graphs[-1].replay()
    torch.cuda.synchronize(device)
    assert all(int(t.abs().sum()) == 0 for t in beyond)


def test_capture_reserve_covers_twice_the_largest_capture(fresh_counters, monkeypatch):
    """``reserve_launch_counters`` fills the reserve before any eager call; a
    capture that takes several pairs makes the next refill keep at least twice
    that many."""
    device = _require_cuda()
    _unhomed(monkeypatch)
    monkeypatch.setattr(cl, "CAPTURE_RESERVE", 4)
    cl.reserve_launch_counters(2, device)
    reserve = cl._COUNTER_STATES[device].reserves[2]
    assert (len(reserve.free), reserve.made) == (4, 4)
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        before = _allocations(device)
        pairs = [cl.launch_counters(2, device) for _ in range(3)]
        assert _allocations(device) == before
    assert (reserve.peak, reserve.issued, len(reserve.free)) == (3, 3, 1)
    cl.launch_counters(2, device)
    assert len(reserve.free) == 6 and reserve.made == 9
    graph.replay()
    torch.cuda.synchronize(device)
    assert all(int(t.abs().sum()) == 0 for pair in pairs for t in pair)
    # Under capture it does nothing.
    work = torch.zeros(1, device=f"cuda:{device}")
    other = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(other, stream=stream):
        cl.reserve_launch_counters(2, device)
        work.add_(1)
    assert reserve.made == 9


def test_sm_table_of_an_invalid_device_is_none():
    _require_cuda()
    assert cl.sm_table(torch.cuda.device_count() + 7) is None
