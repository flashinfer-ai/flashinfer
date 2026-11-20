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
import json
import logging
import subprocess
import sys
import threading
from typing import Any

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


def _fake_device(monkeypatch, probe, clocks=(4752, 4752)):
    # Capture detection is stubbed at the module level: the real one selects the
    # (fake) device first when more than one GPU is visible.
    monkeypatch.setattr(cl, "_is_locality_arch", lambda index: True)
    monkeypatch.setattr(cl, "_stream_capturing", lambda device_index: False)
    monkeypatch.setattr(cl, "_probe", lambda device_index: probe(device_index))
    reads = []

    def memory_clocks(device_index):
        reads.append(device_index)
        return clocks

    monkeypatch.setattr(cl, "_memory_clocks", memory_clocks)
    return reads


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


@pytest.mark.parametrize(
    "clocks, routes",
    [
        ((4752, 4752), True),
        ((4515, 4752), True),  # 0.95 x 4752 = 4514.4
        ((4514, 4752), False),
        ((3996, 4752), False),
        ((3186, 4752), False),
        ((0, 0), False),
        (None, False),  # NVML could not read the clocks
    ],
)
def test_memory_at_full_clock(monkeypatch, clocks, routes):
    """memory_at_full_clock is False for a device whose memory runs below
    MEMORY_CLOCK_FRACTION of its maximum clock, or whose clock NVML cannot read;
    the clock is read once and the outcome is cached either way. The SM table
    neither reads the clock nor depends on it."""
    table = cl.SmTable(device_index=5, counts=(106, 106), entries=None)
    probe = _Probes((table, True))
    reads = _fake_device(monkeypatch, probe, clocks)
    try:
        assert cl.sm_table(5) is table
        assert reads == [] and 5 not in cl._MEMORY_CLOCKS
        assert cl.memory_at_full_clock(5) is routes
        assert cl._MEMORY_CLOCKS[5] == clocks
        assert cl.memory_at_full_clock(5) is routes
        assert cl.memory_at_full_clock(5) is routes
        assert cl.sm_table(5) is table and cl._TABLES[5] is table
        assert probe.calls == 1 and reads == [5]
    finally:
        cl._forget()
    assert not cl._MEMORY_CLOCKS


@pytest.mark.parametrize("outcome", [(None, True), (None, False)])
def test_memory_clock_not_read_without_a_table(monkeypatch, outcome):
    """The probe reads no memory clock, whatever its outcome."""
    probe = _Probes(outcome, outcome)
    reads = _fake_device(monkeypatch, probe, (3186, 4752))
    try:
        assert cl.sm_table(5) is None
        assert reads == [] and 5 not in cl._MEMORY_CLOCKS
    finally:
        cl._forget()


class _FakeNvml:
    """A stand-in ``pynvml`` module: ``fail`` names the call that raises."""

    NVML_CLOCK_MEM = 2

    class NVMLError(Exception):
        pass

    def __init__(self, current=3186, peak=4752, fail=None):
        self.current, self.peak, self.fail = current, peak, fail
        self.calls: list[Any] = []

    def _call(self, name, *args):
        self.calls.append((name, *args))
        if name == self.fail:
            raise self.NVMLError(name)

    def nvmlInit(self):
        self._call("init")

    def nvmlShutdown(self):
        self._call("shutdown")

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self._call("handle", uuid)
        return ("handle", uuid)

    def nvmlDeviceGetClockInfo(self, handle, clock):
        self._call("clock", handle, clock)
        return self.current

    def nvmlDeviceGetMaxClockInfo(self, handle, clock):
        self._call("max_clock", handle, clock)
        return self.peak


def test_memory_clocks_from_nvml(monkeypatch):
    """The device is found by its UUID; NVML is shut down after the reads."""
    nvml = _FakeNvml(current=3186, peak=4752)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(cl, "_nvml_uuid", lambda index: f"GPU-uuid-{index}")
    assert cl._memory_clocks(3) == (3186, 4752)
    handle = ("handle", "GPU-uuid-3")
    assert nvml.calls == [
        ("init",),
        ("handle", "GPU-uuid-3"),
        ("clock", handle, nvml.NVML_CLOCK_MEM),
        ("max_clock", handle, nvml.NVML_CLOCK_MEM),
        ("shutdown",),
    ]


@pytest.mark.parametrize("fail", ["init", "handle", "clock", "max_clock", "uuid"])
def test_memory_clocks_nvml_error(monkeypatch, fail):
    """Any NVML failure reads as no clocks (batch DeepGEMM FP8 then routes no
    twin); NVML is shut down when it was initialized."""
    nvml = _FakeNvml(fail=fail)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)

    def uuid(index):
        if fail == "uuid":
            raise RuntimeError("no device properties")
        return "GPU-uuid"

    monkeypatch.setattr(cl, "_nvml_uuid", uuid)
    assert cl._memory_clocks(0) is None
    names = [call[0] for call in nvml.calls]
    initialized = fail not in ("init", "uuid")
    assert ("shutdown" in names) == initialized


def test_memory_clocks_without_pynvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises ImportError
    assert cl._memory_clocks(0) is None


@pytest.mark.parametrize("fail", [None, "clock"])
def test_unreadable_memory_clock_logged_once_per_device(monkeypatch, caplog, fail):
    """A missing pynvml (``fail`` None) or an NVML failure logs one warning per
    device and process, however often the clock is read; a readable clock logs
    nothing."""
    if fail is None:
        monkeypatch.setitem(sys.modules, "pynvml", None)
    else:
        monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml(fail=fail))
    monkeypatch.setattr(cl, "_nvml_uuid", lambda index: "GPU-uuid")
    monkeypatch.setattr(cl, "_CLOCK_WARNED", set())
    with caplog.at_level(logging.WARNING, logger=cl.__name__):
        for device in (0, 0, 1, 0, 1):
            assert cl._memory_clocks(device) is None
    warnings = [r for r in caplog.records if r.name == cl.__name__]
    assert [r.args[0] for r in warnings] == [0, 1]
    reason = "pynvml is not installed" if fail is None else "NVML could not read it"
    assert all(reason in r.getMessage() for r in warnings)
    caplog.clear()
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml())
    with caplog.at_level(logging.WARNING, logger=cl.__name__):
        assert cl._memory_clocks(2) == (3186, 4752)
    assert not [r for r in caplog.records if r.name == cl.__name__]


def test_memory_at_full_clock_through_nvml(monkeypatch):
    """End to end over a stand-in NVML: below the full clock
    memory_at_full_clock is False and cached, the table stays; NVML is read
    once."""
    table = cl.SmTable(device_index=5, counts=(106, 106), entries=None)
    probe = _Probes((table, True))
    # The stubs of _fake_device without its stand-in clock reader.
    monkeypatch.setattr(cl, "_is_locality_arch", lambda index: True)
    monkeypatch.setattr(cl, "_stream_capturing", lambda device_index: False)
    monkeypatch.setattr(cl, "_probe", lambda device_index: probe(device_index))
    nvml = _FakeNvml(current=3186, peak=4752)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(cl, "_nvml_uuid", lambda index: "GPU-uuid")
    try:
        assert cl.sm_table(5) is table
        assert cl.memory_at_full_clock(5) is False
        assert cl.memory_at_full_clock(5) is False
        assert cl.sm_table(5) is table
        assert probe.calls == 1
        assert [call[0] for call in nvml.calls].count("clock") == 1
        assert cl._MEMORY_CLOCKS[5] == (3186, 4752)
    finally:
        cl._forget()


@pytest.mark.parametrize(
    "uuid, expected",
    [
        (
            "1c2d3e4f-0000-1111-2222-333344445555",
            "GPU-1c2d3e4f-0000-1111-2222-333344445555",
        ),
        ("GPU-1c2d3e4f", "GPU-1c2d3e4f"),
        ("MIG-1c2d3e4f", "MIG-1c2d3e4f"),
    ],
)
def test_nvml_uuid(monkeypatch, uuid, expected):
    class Properties:
        pass

    properties = Properties()
    properties.uuid = uuid
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
    assert cl._nvml_uuid(0) == expected


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


def test_sm_table_on_the_cpu_under_a_cuda_default_device():
    """The SM table is a CPU tensor (the twin bindings take it from host memory)
    also when the default device is CUDA; split_empty allocates nothing else
    there."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    previous = torch.get_default_device()
    cl._forget()
    try:
        torch.set_default_device("cuda")
        table = cl._force_table((106, 106), device_index=7)
        assert table.entries.device.type == "cpu"
        assert table.entries.dtype == torch.int32 and table.entries.is_contiguous()
        assert table.entries.shape == (cl.SM_TABLE_LENGTH,)
        cl._forget()
        device = torch.cuda.current_device()
        if torch.cuda.get_device_capability(device) == (10, 7):
            torch.set_default_device(previous)
            device = _require_locality()
            torch.set_default_device("cuda")
            table = cl.sm_table(device)
            assert table is not None
            assert table.entries.device.type == "cpu"
            assert table.entries.dtype == torch.int32 and table.entries.is_contiguous()
            unit = cl.granularity(device)
            before = _allocations(device)
            tensor = cl.split_empty((2 * unit,), torch.uint8, unit, device)
            assert _allocations(device) == before  # no stray caching-allocator block
            assert cl.homing(tensor) is not None
    finally:
        torch.set_default_device(previous)
        cl._forget()


def test_generated_source_is_published_whole(tmp_path):
    """Generated sources are written to a temporary file and moved into place; an
    unchanged source is not rewritten (its timestamp stays). The file gets the
    mode a plain open() would: 0o666 less the umask."""
    import os
    import stat

    from flashinfer.jit.cake_locality import _publish_if_different

    path = tmp_path / "gen" / "cake_locality_probe.cu"
    previous = os.umask(0o027)
    try:
        _publish_if_different(path, "first")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640
    assert path.read_text() == "first"
    stamp = os.stat(path).st_mtime_ns
    inode = os.stat(path).st_ino
    _publish_if_different(path, "first")
    assert os.stat(path).st_mtime_ns == stamp and os.stat(path).st_ino == inode
    _publish_if_different(path, "second")
    assert path.read_text() == "second"
    assert os.stat(path).st_ino != inode  # replaced, not truncated in place
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


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


def test_memory_at_full_clock_on_the_device():
    """On the device: NVML reads the memory clock, memory_at_full_clock is True
    exactly when that clock is at least MEMORY_CLOCK_FRACTION of the maximum, and
    the SM table exists either way."""
    device = _require_locality()
    cl._forget()
    try:
        table = cl.sm_table(device)
        assert table is not None
        full = cl.memory_at_full_clock(device)
        clocks = cl._MEMORY_CLOCKS.get(device)
        print(f"MEMORY_CLOCKS device={device} clocks={clocks} full={full}")
        if clocks is None:
            pytest.skip("NVML could not read the memory clock")
        current, peak = clocks
        assert peak > 0
        assert full == (current >= cl.MEMORY_CLOCK_FRACTION * peak)
        assert cl.sm_table(device) is table
    finally:
        cl._forget()


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
    # The returned tensor's storage carries its record: every alias starting at
    # the base sees it (callers compare the size for the whole allocation); a
    # view starting past the base and a copy do not.
    assert cl.tensor_homing(tensor) == home
    for alias in (
        tensor[:],
        tensor.detach(),
        tensor.data,
        torch.nn.Parameter(tensor, requires_grad=False),
        tensor[:7],
    ):
        assert cl.tensor_homing(alias) == home
    assert cl.tensor_homing(tensor[5:]) is None
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


def _driver():
    try:
        from cuda.bindings import driver
    except ImportError:
        pytest.skip("the cuda-python driver bindings are required")
    return driver


def _ok(result):
    """The value of a cuda.bindings call ``(error, value)``; fails on an error."""
    assert int(result[0]) == 0, result[0]
    return result[1] if len(result) > 1 else None


def _current_context(driver) -> int:
    return int(_ok(driver.cuCtxGetCurrent()))


def _create_context(driver, device: int):
    """A new (non-primary) context of ``device``, made current."""
    cu_device = _ok(driver.cuDeviceGet(device))
    try:
        return _ok(driver.cuCtxCreate(None, 0, cu_device))
    except TypeError:  # cuda-python 12: cuCtxCreate(flags, dev)
        return _ok(driver.cuCtxCreate(0, cu_device))


def _free_bytes(driver) -> int:
    """Free device memory of the current context's device, from the driver."""
    return int(_ok(driver.cuMemGetInfo()))


# Large enough that a leak is far outside the noise of context creation.
_LEAK_PROBE_BYTES = 512 * 1024**2
_FREE_SLACK = 64 * 1024**2


@pytest.mark.parametrize(
    "case",
    [
        "freed_under_another",
        "made_under_another",
        "context_destroyed",
        "context_destroyed_new_current",
    ],
)
def test_split_empty_freed_in_its_own_context(case):
    """A split allocation is unmapped and leaves the registry when its last tensor
    is freed while another context of the device is current, in the context that
    made it (the caller's context is current again afterwards). When that
    context was destroyed, its memory is still mapped (VMM memory outlives the
    context), and the free unmaps and releases it in the current context, also
    when that is a context made after the destroy, which may reuse the destroyed
    one's handle. The device's free memory returns in every case."""
    device = _require_locality()
    driver = _driver()
    unit = cl.granularity(device)
    nbytes = -(-_LEAK_PROBE_BYTES // unit) * unit
    cut = nbytes // 2 // unit * unit
    gc.collect()
    torch.cuda.synchronize(device)
    primary = _current_context(driver)
    free_before = _free_bytes(driver)
    tensor = None
    if case == "freed_under_another":
        tensor = cl.split_empty((nbytes,), torch.uint8, cut, device)
    other = _create_context(driver, device)
    contexts = [other]  # the live contexts this test made, innermost last
    try:
        assert _current_context(driver) == int(other) != primary
        if tensor is None:
            tensor = cl.split_empty((nbytes,), torch.uint8, cut, device)
            _ok(driver.cuCtxPopCurrent())
            assert _current_context(driver) == primary
            if case.startswith("context_destroyed"):
                _ok(driver.cuCtxDestroy(contexts.pop()))
                # The range outlived its context: its memory is still in use.
                assert _free_bytes(driver) <= free_before - nbytes + _FREE_SLACK
                if case == "context_destroyed_new_current":
                    contexts.append(_create_context(driver, device))
        base = tensor.data_ptr()
        assert cl.homing(base) is not None
        caller = _current_context(driver)
        del tensor
        gc.collect()
        assert cl.homing(base) is None
        assert _current_context(driver) == caller
    finally:
        for context in reversed(contexts):
            if _current_context(driver) == int(context):
                _ok(driver.cuCtxPopCurrent())
            _ok(driver.cuCtxDestroy(context))
    assert _current_context(driver) == primary
    torch.cuda.synchronize(device)
    assert _free_bytes(driver) >= free_before - _FREE_SLACK


# Work queued in a context for far longer than a split free takes: copies of
# _BUSY_BYTES, _BUSY_COPIES times (tens of milliseconds).
_BUSY_BYTES = 256 * 1024**2
_BUSY_COPIES = 1000


def test_split_free_waits_for_the_freeing_context():
    """A split allocation freed while another context of its device is current
    is freed only after the work queued in that context is done (that work may
    read the memory), not only the work of the context that made it."""
    device = _require_locality()
    driver = _driver()
    unit = cl.granularity(device)
    tensor = cl.split_empty((2 * unit,), torch.uint8, unit, device)
    base = tensor.data_ptr()
    torch.cuda.synchronize(device)
    primary = _current_context(driver)
    other = _create_context(driver, device)
    buffers = []
    stream = None
    try:
        assert _current_context(driver) == int(other) != primary
        buffers = [_ok(driver.cuMemAlloc(_BUSY_BYTES)) for _ in range(2)]
        stream = _ok(driver.cuStreamCreate(0))
        for _ in range(_BUSY_COPIES):
            _ok(driver.cuMemcpyDtoDAsync(buffers[1], buffers[0], _BUSY_BYTES, stream))
        busy = driver.cuStreamQuery(stream)[0]
        del tensor  # the last reference: freed now, under `other`
        done = driver.cuStreamQuery(stream)[0]
        assert cl.homing(base) is None
        assert _current_context(driver) == int(other)
        assert busy == driver.CUresult.CUDA_ERROR_NOT_READY, busy
        assert done == driver.CUresult.CUDA_SUCCESS, done
    finally:
        if _current_context(driver) == int(other):
            if stream is not None:
                _ok(driver.cuStreamSynchronize(stream))
                _ok(driver.cuStreamDestroy(stream))
            for buffer in buffers:
                _ok(driver.cuMemFree(buffer))
            _ok(driver.cuCtxPopCurrent())
        _ok(driver.cuCtxDestroy(other))
    assert _current_context(driver) == primary


def test_split_calls_on_another_device_keep_the_callers_context():
    """Allocating, querying and freeing split memory of another device leaves the
    caller's context current, also when it is not a primary context."""
    device = _require_locality()
    peer = next(
        (
            index
            for index in range(torch.cuda.device_count())
            if index != device and torch.cuda.get_device_capability(index) == (10, 7)
        ),
        None,
    )
    if peer is None:
        pytest.skip("needs a second sm_107a device")
    driver = _driver()
    unit = cl.granularity(peer)
    warm = cl.split_empty((2 * unit,), torch.uint8, unit, peer)  # streams, module
    del warm
    gc.collect()
    primary = _current_context(driver)
    other = _create_context(driver, device)
    try:
        assert _current_context(driver) == int(other) != primary
        assert cl.granularity(peer) == unit
        assert _current_context(driver) == int(other)
        tensor = cl.split_empty((2 * unit,), torch.uint8, unit, peer)
        assert _current_context(driver) == int(other)
        base = tensor.data_ptr()
        assert cl.pointer_domain(base + unit, peer) == 1
        assert _current_context(driver) == int(other)
        del tensor
        gc.collect()
        assert cl.homing(base) is None
        assert _current_context(driver) == int(other)
    finally:
        if _current_context(driver) == int(other):
            _ok(driver.cuCtxPopCurrent())
        _ok(driver.cuCtxDestroy(other))
    assert _current_context(driver) == primary


def _runtime_libraries() -> list[Any]:
    """Every CUDA runtime library loaded in this process (the selected device is
    kept per runtime library and thread)."""
    import ctypes

    with open("/proc/self/maps") as maps:
        paths = sorted(
            {
                line.split()[-1]
                for line in maps
                if "/libcudart.so" in line and line.split()[-1].startswith("/")
            }
        )
    return [ctypes.CDLL(path) for path in paths]


def _thread_state(driver, runtimes: list[Any]) -> dict[str, Any]:
    """The calling thread's current context (0: none), the device each runtime
    library reports for it, and ``torch.cuda.current_device()``."""
    import ctypes

    devices = []
    for runtime in runtimes:
        index = ctypes.c_int(-1)
        assert runtime.cudaGetDevice(ctypes.byref(index)) == 0
        devices.append(index.value)
    return {
        "context": _current_context(driver),
        "runtime_devices": devices,
        "torch_device": torch.cuda.current_device(),
    }


def _free_in_a_new_thread(device: int) -> dict[str, Any]:
    """Makes a split allocation on ``device``, then frees it and reads the
    granularity of ``device`` in a new thread that never used CUDA, and then
    allocates ``torch.empty(4, device=f"cuda:{device}")`` in that thread. Returns
    the thread's state (see ``_thread_state``) before and after the split calls,
    whether device 0's primary context is active before and after, and the device
    that holds the memory of the thread's tensor."""
    from cuda.bindings import driver

    _ok(driver.cuInit(0))
    first = _ok(driver.cuDeviceGet(0))

    def primary_active() -> bool:
        error, _flags, active = driver.cuDevicePrimaryCtxGetState(first)
        assert int(error) == 0, error
        return bool(active)

    seen: dict[str, Any] = {"primary_active_at_start": primary_active()}
    unit = cl.granularity(device)
    held = [cl.split_empty((2 * unit,), torch.uint8, unit, device)]
    base = held[0].data_ptr()
    runtimes = _runtime_libraries()
    seen["runtime_libraries"] = len(runtimes)
    # The thread's tensor then needs a new device allocation (no cached block).
    torch.cuda.empty_cache()
    tensors = []

    def run() -> None:
        try:
            seen["before"] = _thread_state(driver, runtimes)
            seen["primary_active_before"] = primary_active()
            del held[0]  # the last reference: the split is freed in this thread
            gc.collect()
            seen["freed"] = cl.homing(base) is None
            seen["after_free"] = _thread_state(driver, runtimes)
            seen["granularity"] = cl.granularity(device)
            seen["after"] = _thread_state(driver, runtimes)
            seen["primary_active_after"] = primary_active()
            tensors.append(torch.empty(4, device=f"cuda:{device}"))
            seen["tensor_device"] = tensors[0].device.index
            seen["primary_active_after_tensor"] = primary_active()
        except BaseException as exc:  # reported by the caller
            seen["error"] = repr(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if tensors:
        seen["tensor_memory_device"] = int(
            _ok(
                driver.cuPointerGetAttribute(
                    driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                    tensors[0].data_ptr(),
                )
            )
        )
    seen["unit"] = unit
    return seen


@pytest.mark.parametrize("process", ["this_process", "new_process"])
def test_split_calls_in_a_thread_without_a_context_keep_its_state(process):
    """Freeing split memory of the last device, and then reading its granularity,
    from a thread with no current context leaves that thread as it was: no current
    context, and the same device reported by the CUDA runtime and by
    ``torch.cuda.current_device()``. A tensor the thread then allocates on the last
    device is in that device's memory, and no primary context is created on device
    0. The new process uses only the last device, so device 0's primary context is
    not active before and that check is not vacuous."""
    _require_locality()
    _driver()
    last = torch.cuda.device_count() - 1
    if last < 1 or torch.cuda.get_device_capability(last) != (10, 7):
        pytest.skip("needs a second sm_107a device")
    if process == "this_process":
        seen = _free_in_a_new_thread(last)
    else:
        script = (
            "import importlib.util, json, sys\n"
            "import torch\n"
            "torch.cuda.set_device(int(sys.argv[2]))\n"
            "spec = importlib.util.spec_from_file_location('locality_tests', sys.argv[1])\n"
            "tests = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(tests)\n"
            "print('SEEN', json.dumps(tests._free_in_a_new_thread(int(sys.argv[2]))))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, __file__, str(last)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        lines = [
            line for line in result.stdout.splitlines() if line.startswith("SEEN ")
        ]
        assert result.returncode == 0 and lines, result.stdout + result.stderr
        seen = json.loads(lines[-1][len("SEEN ") :])
    assert "error" not in seen, seen
    if process == "new_process":
        assert not seen["primary_active_before"], seen
    assert seen["freed"] and seen["granularity"] == seen["unit"], seen
    assert seen["before"]["context"] == 0, seen
    assert seen["after_free"] == seen["before"], seen
    assert seen["after"] == seen["before"], seen
    assert seen["tensor_device"] == last, seen
    assert seen["tensor_memory_device"] == last, seen
    if not seen["primary_active_before"]:
        assert not seen["primary_active_after"], seen
        assert not seen["primary_active_after_tensor"], seen


def _peer_device(device: int) -> int:
    """Another sm_107a device than ``device``; skips the test without one."""
    peer = next(
        (
            index
            for index in range(torch.cuda.device_count())
            if index != device and torch.cuda.get_device_capability(index) == (10, 7)
        ),
        None,
    )
    if peer is None:
        pytest.skip("needs a second sm_107a device")
    return peer


def _raw_base(raw: Any) -> int:
    """The device address of a tensor the locality module returned, read without
    PyTorch (which may select a device)."""
    if isinstance(raw, torch.Tensor):
        return int(raw.data_ptr())
    import ctypes

    capsule = raw._to_dlpack()
    get_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
        ("PyCapsule_GetPointer", ctypes.pythonapi)
    )
    # A DLManagedTensor starts with its DLTensor, which starts with the data
    # pointer.
    return int(ctypes.c_void_p.from_address(get_pointer(capsule, b"dltensor")).value)


def _memory_device(driver, pointer: int) -> int:
    return int(
        _ok(
            driver.cuPointerGetAttribute(
                driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pointer
            )
        )
    )


def test_split_calls_under_a_pushed_context_of_another_device_keep_its_state():
    """A thread that pushes its own context of one device, allocates, queries and
    frees split memory of another device through the locality module, and pops
    its context is left as it was: no current context, and the same device
    reported by the CUDA runtime and by ``torch.cuda.current_device()``. A tensor
    the thread then allocates on the other device is in that device's memory."""
    device = _require_locality()
    driver = _driver()
    peer = _peer_device(device)
    module = cl._module()
    unit = cl.granularity(peer)
    warm = cl.split_empty((2 * unit,), torch.uint8, unit, peer)  # streams, module
    del warm
    gc.collect()
    runtimes = _runtime_libraries()
    # The thread's tensor then needs a new device allocation (no cached block).
    torch.cuda.empty_cache()
    seen: dict[str, Any] = {}
    tensors = []

    def run() -> None:
        try:
            seen["before"] = _thread_state(driver, runtimes)
            context = _create_context(driver, device)
            try:
                seen["pushed"] = int(context) == _current_context(driver)
                seen["granularity"] = int(module.cake_locality_granularity(peer))
                raw = module.cake_locality_split_empty(2 * unit, unit, peer)
                base = _raw_base(raw)
                seen["domains"] = [
                    int(module.cake_locality_pointer_domain(base + offset, peer))
                    for offset in (0, unit)
                ]
                seen["registered"] = cl.homing(base) is not None
                del raw  # the last reference: freed now, under the pushed context
                seen["freed"] = cl.homing(base) is None
                seen["kept"] = int(context) == _current_context(driver)
            finally:
                _ok(driver.cuCtxPopCurrent())
                _ok(driver.cuCtxDestroy(context))
            seen["after"] = _thread_state(driver, runtimes)
            tensors.append(torch.empty(4, device=f"cuda:{peer}"))
            seen["tensor_device"] = tensors[0].device.index
        except BaseException as exc:  # reported below
            seen["error"] = repr(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert "error" not in seen, seen
    assert seen["before"]["context"] == 0, seen
    assert seen["pushed"] and seen["kept"], seen
    assert seen["granularity"] == unit and seen["domains"] == [0, 1], seen
    assert seen["registered"] and seen["freed"], seen
    assert seen["after"] == seen["before"], seen
    assert seen["tensor_device"] == peer, seen
    assert _memory_device(driver, tensors[0].data_ptr()) == peer, seen


def test_split_allocation_in_a_thread_without_a_context_keeps_its_state():
    """A split allocation of the last device made through the locality module in
    a thread with no current context leaves that thread as it was (no current
    context, the same runtime and PyTorch device), and its memory is on that
    device, each part on its die."""
    device = _require_locality()
    driver = _driver()
    last = torch.cuda.device_count() - 1
    if last < 1 or torch.cuda.get_device_capability(last) != (10, 7):
        pytest.skip("needs a second sm_107a device")
    torch.cuda.synchronize(device)  # this thread's context reads the pointers below
    module = cl._module()
    unit = cl.granularity(last)
    runtimes = _runtime_libraries()
    seen: dict[str, Any] = {}
    held = []

    def run() -> None:
        try:
            seen["before"] = _thread_state(driver, runtimes)
            held.append(module.cake_locality_split_empty(2 * unit, unit, last))
            seen["after_alloc"] = _thread_state(driver, runtimes)
            base = _raw_base(held[0])
            seen["domains"] = [
                int(module.cake_locality_pointer_domain(base + offset, last))
                for offset in (0, unit)
            ]
            seen["after"] = _thread_state(driver, runtimes)
            seen["base"] = base
        except BaseException as exc:  # reported below
            seen["error"] = repr(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert "error" not in seen, seen
    assert seen["before"]["context"] == 0, seen
    assert seen["after_alloc"] == seen["before"], seen
    assert seen["after"] == seen["before"], seen
    assert seen["domains"] == [0, 1], seen
    base = seen["base"]
    record = cl.homing(base)
    assert record is not None and record.device_index == last, record
    assert _memory_device(driver, base) == last
    assert _memory_device(driver, base + unit) == last
    held.clear()
    gc.collect()
    assert cl.homing(base) is None


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


# The message ``_domain_pools`` raises when the pools' first allocation fails: the
# allocator's error follows the colon, and the cause is a torch.OutOfMemoryError
# whatever the driver error was. Without the colon part, PyTorch refused the
# request before the allocator ran (for example at its per-process memory limit).
_SMOKE_FAILURE_NO_DETAIL = "locality-domain 0 allocation smoke test failed"
_SMOKE_FAILURE = (
    _SMOKE_FAILURE_NO_DETAIL + ": localized VMM allocation in domain 0 failed: {}"
)


@pytest.mark.parametrize(
    "case", ["direct", "wrapped", "wrapped_no_diagnostic", "wrapped_refusal"]
)
def test_launch_counters_out_of_memory_pair_not_cached(
    fresh_counters, monkeypatch, case
):
    """A pair made unhomed because a locality-domain pool was out of memory serves
    its call and is not cached; the next call tries the pools again. The same
    holds when the pools' first allocation wraps a CUDA out-of-memory error in a
    LocalityDomainError (as ``_domain_pools`` does), also one with no allocator
    error (PyTorch refused the request). A wrapped driver refusal
    (CUDA_ERROR_NOT_SUPPORTED), like any other refusal, is remembered."""
    from flashinfer import locality_domain

    device = _require_cuda()
    calls = []
    message = {
        "wrapped": _SMOKE_FAILURE.format("CUDA_ERROR_OUT_OF_MEMORY (2): out of memory"),
        "wrapped_no_diagnostic": _SMOKE_FAILURE_NO_DETAIL,
        "wrapped_refusal": _SMOKE_FAILURE.format(
            "CUDA_ERROR_NOT_SUPPORTED (801): operation not supported"
        ),
    }.get(case)

    def exhausted(_index):
        calls.append(_index)
        if message is None:
            raise torch.OutOfMemoryError("pool exhausted")
        try:
            raise torch.OutOfMemoryError("pool exhausted")
        except torch.OutOfMemoryError as exc:
            raise locality_domain.LocalityDomainError(message) from exc

    monkeypatch.setattr(locality_domain, "_domain_pools", exhausted)
    stream = torch.cuda.Stream(device=device)
    if case == "wrapped_refusal":
        with torch.cuda.stream(stream):
            first = cl.launch_counters(2, device)
            assert cl.launch_counters(2, device)[0] is first[0]
        torch.cuda.synchronize(device)
        assert all(int(t.abs().sum()) == 0 for t in first)
        assert calls == [device] and device in cl._POOLS_UNAVAILABLE
        return
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
        assert cl._stream_capturing(device)
        assert cl._capture_id(device) != 0  # None without CUDA Python bindings
    assert not cl._stream_capturing(device)
    assert cl._capture_id(device) in (0, None)
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


def test_capture_reserve_counts_each_capture_separately(fresh_counters, monkeypatch):
    """Overlapping captures keep separate counts: capture A takes 34 pairs, B one,
    then A 21 more, so the most pairs one capture took is 55 and the next refill
    keeps 110 pairs. Only the most recent _CAPTURE_IDS_TRACKED capture ids are
    kept."""
    device = _require_cuda()
    _unhomed(monkeypatch)
    current = []
    monkeypatch.setattr(cl, "_capture_id", lambda _index: current[-1])
    for capture, count in ((101, 34), (102, 1), (101, 21)):
        current.append(capture)
        for _ in range(count):
            cl._captured_pair(device, 2)
    reserve = cl._COUNTER_STATES[device].reserves[2]
    assert reserve.peak == 55
    assert list(reserve.taken.items()) == [(102, 1), (101, 55)]
    cl.reserve_launch_counters(2, device)
    assert len(reserve.free) == 2 * 55
    tracked = cl._CAPTURE_IDS_TRACKED
    for capture in range(1000, 1000 + 2 * tracked):
        current.append(capture)
        cl._captured_pair(device, 2)
    assert list(reserve.taken) == list(range(1000 + tracked, 1000 + 2 * tracked))
    assert reserve.peak == 55


def test_sm_table_of_an_invalid_device_is_none():
    _require_cuda()
    assert cl.sm_table(torch.cuda.device_count() + 7) is None
