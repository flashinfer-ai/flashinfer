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
