# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only tests of the locality-domain twin of the Cake nontranspose FP32 T=1
decode on a dual-die SM107a GPU: its records and host template, its route table,
the dispatch of the public nontranspose path (twin or whole-GPU launch, with fake
CUDA tensors and kernel entries) and the refusals of
``localize_gdn_decode_state``."""

from __future__ import annotations

import re
import threading
import types

import pytest
import torch

import flashinfer.cake_gdn_decode_locality as locality
import flashinfer.gdn_decode as gdn_decode
from flashinfer import cake_locality
from flashinfer.jit import cake_gdn

pytestmark = pytest.mark.cpu_only

_WHOLE = "gdn_decode_nontranspose_fp32_t1_hqsplit_tma3"
_TWIN = "gdn_decode_nontranspose_fp32_t1_hqsplit_tma3_domain"
_TEMPLATE = "host/templates/cake_decode_fp32_t1_tma_state_domain.cc.in"
_SCALE = 128**-0.5
_ROW_BYTES = 32 * 128 * 128 * 4
_BATCH_CLASSES = ("B48-63", "B64-127", "B128-255", "B>=256")


def _records(schedule: str) -> list[dict]:
    return [
        record
        for record in cake_gdn._manifest()["variants"]
        if record["source_schedule"]
        == f"flashinfer.cake_gdn.generated.decode:{schedule}"
    ]


def _twin_name(scale: float) -> str:
    (record,) = [
        record
        for record in _records(_TWIN)
        if record["specializations"]["SCALE"] == scale
    ]
    return record["name"]


def _whole_name(scale: float) -> str:
    (record,) = [
        record
        for record in _records(_WHOLE)
        if record["specializations"]["SCALE"] == scale
    ]
    return record["name"]


def _kernel_params(source: str, symbol: str) -> list[str]:
    """The parameter names of ``symbol`` in a kernel unit, in order."""
    match = re.search(rf"\b{symbol}\(([^)]*)\)", source)
    assert match is not None, symbol
    flat = re.sub(r"<[^<>]*>", "<>", match.group(1))
    return [re.findall(r"(\w+)\s*$", param.strip())[0] for param in flat.split(",")]


def _shim_kargs(text: str) -> list[str]:
    """The kernel argument names of a rendered host shim's ``kargs`` list, in order."""
    match = re.search(r"void\* kargs\[\] = \{([^}]*)\};", text)
    assert match is not None
    names = []
    for item in match.group(1).split(","):
        item = item.strip()
        assert item.startswith("&"), item
        names.append(re.sub(r"^(p|v)_", "", item[1:]))
    return names


# --------------------------------------------------------------------------- records


def test_twin_records_launch_through_the_domain_template() -> None:
    # Two records (H16 / HV32 at both scales) share one sm_107a unit, spell the
    # grid-constant tensor-map ABI and launch through the twin's TMA-state template
    # with the whole body's dynamic shared memory; the twin is a TMA-state schedule
    # and not a per-state-head grid (no CTAS_PER_STATE entry).
    twins = _records(_TWIN)
    assert len(twins) == 2
    assert {record["specializations"]["SCALE"] for record in twins} == {_SCALE, 1.0}
    assert len({record["outputs"][0]["path"] for record in twins}) == 1
    for record in twins:
        assert record["domain"] == "decode"
        assert record["tma_abi"] == "grid_constant"
        assert record["architectures"] == ["sm_107a"]
        assert len(record["outputs"]) == 1
        assert record["specializations"]["H"] == 16
        assert record["specializations"]["HV"] == 32
        assert "defines" not in record
        assert record.get("compile_options", ["--use_fast_math"]) == ["--use_fast_math"]
        host = record["host_binding"]
        assert host["template"] == _TEMPLATE
        assert host["cuda_symbol"] == f"kernel_{_TWIN}"
        assert host["params"] == {
            "DYNAMIC_SMEM_BYTES": 51328,
            "H": 16,
            "HV": 32,
            "HV_MINUS_1": 31,
        }
    assert _TWIN in cake_gdn.CAKE_GDN_DECODE_TMA_STATE_SCHEDULES
    assert _TWIN not in cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_CTAS_PER_STATE
    assert _TEMPLATE in cake_gdn._DECODE_TMA_TEMPLATES
    with pytest.raises(cake_gdn.CakeGDNUnsupportedError, match="not an fp32-state"):
        cake_gdn.cake_gdn_nontranspose_fp32_t1_ctas_per_state(_twin_name(_SCALE))
    # The twin's records keep the whole body's specializations, so each whole record
    # has exactly one twin.
    assert sorted(r["specializations"]["SCALE"] for r in _records(_WHOLE)) == sorted(
        r["specializations"]["SCALE"] for r in twins
    )


def test_twin_shim_passes_the_kernel_arguments_in_the_kernel_order() -> None:
    # The rendered shim's kernel arguments are the twin kernel's parameters in order:
    # the whole body's, then the batch size, rows0, the two claim counters and the SM
    # table (by value).  The whole body's shim passes the whole kernel's parameters.
    root = cake_gdn._source_dir()
    twin_extra = ["q_size_p0", "rows0", "counters0", "counters1", "sm_table"]
    for schedule in (_WHOLE, _TWIN):
        for record in _records(schedule):
            source = (root / record["outputs"][0]["path"]).read_text(encoding="utf-8")
            params = _kernel_params(source, record["host_binding"]["cuda_symbol"])
            text = cake_gdn.render_cake_gdn_host_shim(record)
            assert _shim_kargs(text) == params, record["name"]
            if schedule == _TWIN:
                assert params[-len(twin_extra) :] == twin_extra
            else:
                assert not set(twin_extra) & set(params)
    record = _records(_TWIN)[0]
    source = (root / record["outputs"][0]["path"]).read_text(encoding="utf-8")
    assert re.search(
        r"long long q_size_p0, int rows0, int\* __restrict__ counters0, "
        r"int\* __restrict__ counters1, const CakeParamArray<int, 256> sm_table\)",
        source,
    )


def test_twin_shim_checks_its_arguments_and_caps_the_grid_by_residency() -> None:
    # The twin's shim keeps the TMA-state checks and map cache, checks rows0, the
    # counters and the SM table, accepts any grid of 1 .. B * HV CTAs and launches at
    # most the resident CTAs, from an occupancy query made after the dynamic
    # shared-memory opt-in (4 CTAs per SM without one).
    for record in _records(_TWIN):
        text = cake_gdn.render_cake_gdn_host_shim(record)
        assert "@" not in text
        assert "thread_local StateTmaCacheEntry cache[kStateTmaCacheSlots]" in text
        assert "StateTma4D(arg_state, (int)dev.device_id)" in text
        assert "16-byte aligned for the TMA state path" in text
        assert (
            "int64_t rows0, TensorView arg_counters0, TensorView arg_counters1, "
            "TensorView arg_sm_table, int64_t grid_x, int64_t grid_y, int64_t grid_z)"
        ) in text
        assert "rows0 >= 0 && rows0 <= arg_q.size(0)" in text
        assert "grid_x > 0 && grid_x <= arg_q.size(0) * 32LL" in text
        assert "arg_q.size(0) <= 2147483647LL / 32LL" in text
        for name in ("counters0", "counters1"):
            assert f'CheckCudaTensor(arg_{name}, "{name}");' in text
            assert f'CheckDtype(arg_{name}, "{name}", 0, 32, 1);' in text
            assert f'CheckContiguous(arg_{name}, "{name}");' in text
            assert f'CheckSameCudaDevice(arg_{name}, arg_q, "{name}", "q");' in text
        assert "arg_sm_table.device().device_type == kDLCPU" in text
        assert 'CheckDtype(arg_sm_table, "sm_table", 0, 32, 1);' in text
        assert "arg_sm_table.numel() == kSmTableLength" in text
        assert "std::memcpy(v_sm_table.v, DataPtr(arg_sm_table)" in text
        assert "int32_t v_rows0 = (int32_t)rows0;" in text
        assert "int64_t v_q_size_p0 = arg_q.size(0);" in text
        opt_in = text.index(
            "CakeGDNSetMaxDynamicSmem(kernel, (int)dev.device_id, 51328,"
        )
        resident = text.index(
            "ResidentCtas(kernel, (int)dev.device_id, 51328, cake_gdn_resident_cache"
        )
        assert opt_in < resident
        assert (
            "cuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, function, 256, "
            "(size_t)smem_bytes)"
        ) in text
        assert "per_sm = 4;" in text
        assert "grid_x < resident ? grid_x : resident" in text
        assert (
            "dim3 grid((uint32_t)launch_x, (uint32_t)grid_y, (uint32_t)grid_z)" in text
        )
        assert "kernel.Launch(kargs, grid, block, stream, 51328u)" in text
        assert "CTAS_PER_BATCH_ROW" not in text


# --------------------------------------------------------------------------- routes


def test_twin_routes_mirror_the_cake_program_cells() -> None:
    # 212-SM sm_107a devices route hqsplit_tma3 to its twin in both die classes for
    # every batch class from 48 rows on; no other architecture, SM count, schedule or
    # batch class (B<32 and B32-47 included) is listed.
    cells = {
        (_WHOLE, die, batch_class): _TWIN
        for die in ("even", "uneven")
        for batch_class in _BATCH_CLASSES
    }
    routes = cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_DOMAIN_ROUTES
    assert routes == {("sm_107a", 212): cells}
    batch_class = cake_gdn.cake_gdn_nontranspose_fp32_t1_domain_batch_class
    for batch_size, expected in (
        (1, "B<32"),
        (9, "B<32"),
        (10, "B<32"),
        (31, "B<32"),
        (32, "B32-47"),
        (36, "B32-47"),
        (37, "B32-47"),
        (47, "B32-47"),
        (48, "B48-63"),
        (63, "B48-63"),
        (64, "B64-127"),
        (127, "B64-127"),
        (128, "B128-255"),
        (255, "B128-255"),
        (256, "B>=256"),
        (512, "B>=256"),
        (1024, "B>=256"),
    ):
        assert batch_class(batch_size) == expected, batch_size
    # The die class: even up to a gap of 4 SMs between the dies.
    for counts, expected in (
        ((106, 106), "even"),
        ((108, 104), "even"),
        ((104, 108), "even"),
        ((109, 104), "uneven"),
        ((100, 112), "uneven"),
        ((112, 100), "uneven"),
    ):
        assert cake_locality.die_class(counts) == expected, counts


@pytest.mark.parametrize("scale", [_SCALE, 1.0])
def test_twin_selector_names_the_twin_of_the_listed_cells_only(scale: float) -> None:
    def twin(arch="sm_107a", sm_count=212, die="even", batch_size=64, whole=None):
        return cake_gdn.select_cake_gdn_decode_domain_variant(
            arch=arch,
            whole_variant=whole or _whole_name(scale),
            sm_count=sm_count,
            die_class=die,
            batch_size=batch_size,
        )

    for die in ("even", "uneven"):
        for batch_size in (48, 49, 56, 63, 64, 127, 128, 255, 256, 512, 1024):
            assert twin(die=die, batch_size=batch_size) == _twin_name(scale)
        for batch_size in (1, 9, 10, 21, 31, 32, 36, 37, 44, 47):
            assert twin(die=die, batch_size=batch_size) is None
    # Unlisted SM counts (216-SM boards are untimed) and architectures.
    for sm_count in (216, 211, 213, 148, 160):
        assert twin(sm_count=sm_count) is None
    for arch in ("sm_100a", "sm_103a"):
        assert twin(arch=arch, sm_count=212) is None
    # Only hqsplit_tma3 has a twin: no other nontranspose fp32 T=1 record maps to one.
    for record in cake_gdn._manifest()["variants"]:
        schedule = record["source_schedule"].rsplit(":", 1)[-1]
        if (
            record["domain"] != "decode"
            or not schedule.startswith("gdn_decode_nontranspose_fp32_t1")
            or schedule == _WHOLE
        ):
            continue
        assert twin(whole=record["name"]) is None, record["name"]


def test_nontranspose_domain_variant_follows_the_whole_route() -> None:
    # The row's whole route decides: the twin of hqsplit_tma3 from 48 rows on for
    # H16 / HV32 at the two shipped scales; every other row (another head geometry or
    # scale, no l2norm, the small bands, 32 to 47 rows, B200 / B300, 216 SMs) has none.
    def variant(**overrides):
        row = dict(
            arch="sm_107a",
            sm_count=212,
            die_class="uneven",
            batch_size=64,
            num_q_heads=16,
            num_k_heads=16,
            num_v_heads=32,
            scale=_SCALE,
        )
        row.update(overrides)
        return locality.nontranspose_domain_variant(**row)

    assert variant() == _twin_name(_SCALE)
    assert variant(scale=1.0) == _twin_name(1.0)
    assert variant(batch_size=48, die_class="even") == _twin_name(_SCALE)
    assert variant(batch_size=1024) == _twin_name(_SCALE)
    for overrides in (
        dict(batch_size=47),
        dict(batch_size=47, scale=1.0),
        dict(batch_size=32, die_class="even"),
        dict(batch_size=32, scale=1.0),
        dict(batch_size=31),
        dict(batch_size=9),
        dict(scale=0.5),
        dict(num_q_heads=4, num_k_heads=4),
        dict(num_q_heads=4, num_k_heads=4, num_v_heads=8),
        dict(num_k_heads=8),
        dict(use_qk_l2norm=False),
        dict(sm_count=216),
        dict(arch="sm_100a", sm_count=148),
        dict(arch="sm_103a", sm_count=160),
    ):
        assert variant(**overrides) is None, overrides


# --------------------------------------------------------------------------- dispatch


class _FakeCuda(torch.Tensor):
    """A meta tensor that the nontranspose host checks take for a tensor of cuda:0,
    at base address ``address`` (0 unless set)."""

    address = 0

    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda", 0)

    def get_device(self):
        return 0

    def data_ptr(self):
        return self.address


def _fake(shape, dtype, address=0) -> torch.Tensor:
    tensor = torch.Tensor._make_subclass(
        _FakeCuda, torch.empty(shape, dtype=dtype, device="meta")
    )
    tensor.address = address
    return tensor


class _FakeCudaWrapper(_FakeCuda):
    """A ``__torch_dispatch__`` subclass of :class:`_FakeCuda` (no op may reach it)."""

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise AssertionError(f"dispatched {func} to a wrapper subclass")


def _wrapper(shape, dtype) -> torch.Tensor:
    return torch.Tensor._make_subclass(
        _FakeCudaWrapper, torch.empty(shape, dtype=dtype, device="meta")
    )


class _Converted:
    """What the fake ``_cake_gdn_launch_arg`` makes of ``tensor``: equal to another
    conversion of the same object, and to nothing else, so a twin argument shows
    whether it went through the conversion, and from what."""

    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def __eq__(self, other) -> bool:
        return isinstance(other, _Converted) and other.tensor is self.tensor

    def __repr__(self) -> str:
        return f"_Converted({self.tensor!r})"


def _decode_inputs(batch_size: int) -> dict[str, torch.Tensor]:
    bf16, fp32 = torch.bfloat16, torch.float32
    return {
        "q": _fake((batch_size, 1, 16, 128), bf16),
        "k": _fake((batch_size, 1, 16, 128), bf16),
        "v": _fake((batch_size, 1, 32, 128), bf16),
        "state": _fake((batch_size, 32, 128, 128), fp32),
        "A_log": _fake((32,), fp32),
        "a": _fake((batch_size, 1, 32), bf16),
        "dt_bias": _fake((32,), fp32),
        "b": _fake((batch_size, 1, 32), bf16),
        "output": _fake((batch_size, 1, 32, 128), bf16),
    }


class _Device:
    """Fakes for one device: its capability and SM count, its SM table (counting the
    queries), the state homing records, the current stream (or a capture), the claim
    counters, the conversion of launch arguments (``_Converted``), the warnings and
    the kernel entries (each records its calls; ``launch_error`` fails every one)."""

    def __init__(
        self, monkeypatch, *, capability=(10, 7), sm_count=212, counts=(106, 106)
    ):
        self.calls: list[tuple[str, tuple]] = []
        self.loads: list[str] = []
        self.counter_calls: list[tuple[int, int]] = []
        self.homings: dict[int, cake_locality.Homing] = {}
        self.failing: set[str] = set()
        self.failing_launch: set[str] = set()
        self.launch_error: str | None = None
        self.warnings: list[str] = []
        self.table = self.make_table(counts)
        self.counters = (object(), object())
        self.table_queries = 0
        self.capturing = False
        self.stream = 7
        properties = types.SimpleNamespace(
            major=capability[0], minor=capability[1], multi_processor_count=sm_count
        )
        monkeypatch.setattr(gdn_decode, "get_compute_capability", lambda _d: capability)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _i: properties)
        monkeypatch.setattr(cake_gdn, "load_cake_gdn_kernel", self.load)
        monkeypatch.setattr(cake_locality, "sm_table", self.sm_table)
        monkeypatch.setattr(cake_locality, "tensor_homing", self.tensor_homing)
        monkeypatch.setattr(cake_locality, "launch_counters", self.launch_counters)
        monkeypatch.setattr(gdn_decode, "_cake_gdn_eager_stream", self.eager_stream)
        monkeypatch.setattr(gdn_decode, "_cake_gdn_launch_arg", _Converted)
        monkeypatch.setattr(
            gdn_decode,
            "logger",
            types.SimpleNamespace(
                warning=lambda message, *args: self.warnings.append(message % args)
            ),
        )
        monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_LAUNCHES", {})
        monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES", {})
        monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES", {})
        monkeypatch.setattr(gdn_decode, "_CAKE_GDN_DOMAIN_PAIRS", threading.local())
        monkeypatch.setattr(gdn_decode, "_CAKE_GDN_DOMAIN_CAPTURED", set())

    @staticmethod
    def make_table(counts):
        if counts is None:
            return None
        return cake_locality.SmTable(
            device_index=0,
            counts=tuple(counts),
            entries=torch.zeros(cake_locality.SM_TABLE_LENGTH, dtype=torch.int32),
        )

    def load(self, name, arch, **kwargs):
        assert not kwargs
        self.loads.append(name)
        if name in self.failing:
            raise RuntimeError(f"failed to compile {name}")

        def entry(*args):
            self.calls.append((name, args))
            if self.launch_error is not None:
                raise RuntimeError(f"{self.launch_error}: {name}")
            if name in self.failing_launch:
                raise RuntimeError(f"failed to load the kernel of {name}")

        return entry

    def sm_table(self, device=None):
        self.table_queries += 1
        return self.table

    def tensor_homing(self, tensor):
        return self.homings.get(id(tensor))

    def eager_stream(self, device_index):
        return None if self.capturing else (device_index, self.stream)

    def launch_counters(self, words, device=None):
        self.counter_calls.append((words, device))
        return self.counters

    def home(self, state, *, rows0=None, nbytes=None, device_index=0):
        rows = state.shape[0]
        rows0 = (rows + 1) // 2 if rows0 is None else rows0
        nbytes = state.numel() * 4 if nbytes is None else nbytes
        self.homings[id(state)] = cake_locality.Homing(
            device_index=device_index,
            base=0,
            nbytes=nbytes,
            size=nbytes,
            cut=rows0 * _ROW_BYTES,
        )


def _run(inputs, scale=_SCALE):
    return gdn_decode._run_cake_gdn_decode_nontranspose(
        **inputs, scale=scale, use_qk_l2norm=True
    )


def _whole_args(inputs, grid_x):
    return tuple(
        inputs[name]
        for name in ("q", "k", "v", "state", "A_log", "a", "dt_bias", "b", "output")
    ) + (grid_x, 1, 1)


def _twin_args(inputs, device, rows0, *, captured=False):
    # The SM table converted once by the binding; an eager launch's converted counter
    # pair, a captured launch's pair as launch_counters returned it.
    batch_size = inputs["q"].shape[0]
    counters = device.counters
    if not captured:
        counters = tuple(_Converted(counter) for counter in counters)
    return _whole_args(inputs, 0)[:9] + (
        rows0,
        *counters,
        _Converted(device.table.entries),
        batch_size * 32,
        1,
        1,
    )


def test_unhomed_state_launches_the_whole_route_as_before(monkeypatch) -> None:
    # An unhomed state runs the row's whole launch with its whole grid and takes no
    # counters and no twin launch table entry, on every architecture and batch.
    for capability, sm_count in (
        ((10, 7), 212),
        ((10, 7), 216),
        ((10, 0), 148),
        ((10, 3), 160),
    ):
        device = _Device(monkeypatch, capability=capability, sm_count=sm_count)
        for batch_size in (9, 16, 32, 64, 256):
            inputs = _decode_inputs(batch_size)
            for _ in range(2):
                assert _run(inputs) is inputs["output"]
            (route,) = {name for name, _ in device.calls if name.startswith("decode_")}
            assert _TWIN not in route
            whole = cake_gdn.cake_gdn_nontranspose_fp32_t1_ctas_per_state(route)
            expected = _whole_args(inputs, batch_size * 32 * whole)
            assert device.calls == [(route, expected)] * 2
            device.calls.clear()
        assert device.counter_calls == []
        assert gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES == {}
        assert gdn_decode._CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES == {}
        assert not any(_TWIN in name for name in device.loads)


@pytest.mark.parametrize("backend", [None, "auto", "cake_gdn"])
def test_unaligned_inputs_raise_before_any_launch(monkeypatch, backend) -> None:
    # Contiguous views at an unaligned storage offset: the kernels read q and k in
    # 8-byte vectors and the state in 16-byte copies, so the call raises before it
    # resolves or launches anything, homed or not, with the row's launch cached or
    # not.  The check is row- and GPU-independent: through the public call
    # (``backend`` not None) backend="auto" raises too, also on a device the Cake
    # kernels do not serve; it does not fall back to the FlashInfer kernel.
    device = _Device(monkeypatch)

    def flashinfer_kernel(*args, **kwargs):
        raise AssertionError("fell back to the FlashInfer kernel")

    monkeypatch.setattr(gdn_decode, "run_nontranspose_decode", flashinfer_kernel)

    def run(inputs):
        if backend is None:
            return _run(inputs)
        output, _ = gdn_decode.gated_delta_rule_decode(
            **inputs, scale=_SCALE, backend=backend
        )
        return output

    def run_misaligned(device):
        for homed in (False, True):
            for name, address in (
                ("q", 2),
                ("q", 4),
                ("k", 6),
                ("state", 4),
                ("state", 8),
            ):
                inputs = _decode_inputs(64)
                tensor = inputs[name]
                inputs[name] = _fake(tensor.shape, tensor.dtype, address)
                if homed:
                    device.home(inputs["state"])
                with pytest.raises(ValueError, match="aligned"):
                    run(inputs)

    # 8-byte aligned q and k and a 16-byte aligned state launch as before; the other
    # tensors, read and written one element at a time, need only their element
    # alignment.  This runs first: no state is homed yet, and the row's launch is
    # cached for the misaligned calls that follow.
    inputs = _decode_inputs(64)
    for name, address in (
        ("q", 8),
        ("k", 1032),
        ("state", 16),
        ("v", 2),
        ("a", 2),
        ("b", 6),
        ("output", 2),
        ("A_log", 4),
        ("dt_bias", 12),
    ):
        inputs[name] = _fake(inputs[name].shape, inputs[name].dtype, address)
    assert run(inputs) is inputs["output"]
    ((route, args),) = device.calls
    assert _TWIN not in route and args[:9] == _whole_args(inputs, 0)[:9]
    loads = list(device.loads)
    run_misaligned(device)
    assert len(device.calls) == 1 and device.loads == loads
    # On a device the Cake kernels do not serve (SM90), the misaligned calls raise
    # all the same, before the architecture is looked up.  An aligned call there
    # raises CakeGDNUnsupportedError, which backend="auto" turns into a fallback.
    device = _Device(monkeypatch, capability=(9, 0), sm_count=132)
    run_misaligned(device)
    assert device.calls == [] and device.loads == []
    if backend == "auto":
        unsupported = pytest.raises(AssertionError, match="fell back")
    else:
        unsupported = pytest.raises(
            cake_gdn.CakeGDNUnsupportedError, match=r"compute capability 9\.0"
        )
    with unsupported:
        run(_decode_inputs(64))
    assert device.calls == [] and device.loads == []


@pytest.mark.parametrize("scale", [_SCALE, 1.0])
@pytest.mark.parametrize("counts", [(106, 106), (100, 112)])
def test_homed_state_launches_the_twin_in_a_listed_cell(
    monkeypatch, scale, counts
) -> None:
    # A whole homed allocation on a 212-SM sm_107a device with an SM table runs the
    # twin of its row: rows0 at the state's die cut, the stream's counter pair, the SM
    # table by value and a B * HV grid.  The whole launch table keeps its value; the
    # twin is loaded once, and the stream's counter pair is taken once and reused.
    device = _Device(monkeypatch, counts=counts)
    for batch_size, rows0 in ((48, 24), (63, 31), (64, 13), (512, 256)):
        inputs = _decode_inputs(batch_size)
        device.home(inputs["state"], rows0=rows0)
        loads = device.loads.count(_twin_name(scale))
        for _ in range(3):
            assert _run(inputs, scale) is inputs["output"]
        expected = _twin_args(inputs, device, rows0)
        assert device.calls == [(_twin_name(scale), expected)] * 3
        # one resolution of the row's twin launch for the three calls
        assert device.loads.count(_twin_name(scale)) == loads + 1
        key = (0, batch_size, 16, 16, 32, scale, True)
        whole = gdn_decode._CAKE_GDN_NONTRANSPOSE_LAUNCHES[key]
        assert len(whole) == 2 and whole[1] == batch_size * 32
        twin = gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES[
            key + (tuple(counts), rows0 * _ROW_BYTES)
        ]
        assert twin[1:] == (batch_size * 32, rows0, device.table)
        record = device.homings[id(inputs["state"])]
        assert gdn_decode._CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES[id(record)] == (
            record,
            whole,
            twin,
            _Converted(device.table.entries),
        )
        device.calls.clear()
    assert device.counter_calls == [(locality.DOMAIN_COUNTER_WORDS, 0)]
    assert locality.DOMAIN_COUNTER_WORDS == 1


def test_homed_state_rows0_is_the_row_nearest_the_cut(monkeypatch) -> None:
    device = _Device(monkeypatch)
    inputs = _decode_inputs(64)
    state = inputs["state"]
    for cut, rows0 in (
        (0, 0),
        (13 * _ROW_BYTES, 13),
        (13 * _ROW_BYTES + _ROW_BYTES // 2 - 1, 13),
        (13 * _ROW_BYTES + _ROW_BYTES // 2, 14),
        (64 * _ROW_BYTES, 64),
    ):
        device.homings[id(state)] = cake_locality.Homing(
            device_index=0,
            base=0,
            nbytes=64 * _ROW_BYTES,
            size=64 * _ROW_BYTES,
            cut=cut,
        )
        _run(inputs)
        name, args = device.calls.pop()
        assert name == _twin_name(_SCALE) and args[9] == rows0, cut


def test_homed_state_falls_back_to_the_whole_launch(monkeypatch) -> None:
    # A part of a homed allocation, a homing record of another device, no SM table, a
    # row below 48 (B16 of the small band, B32 / B47 of the large band), an untimed SM
    # count and B200 run the whole launch, bit for bit the unhomed call's; a row
    # without a twin is remembered as such (not resolved again).
    def whole_call(device, inputs):
        _run(inputs)
        ((name, args),) = device.calls
        device.calls.clear()
        assert _TWIN not in name
        return args

    device = _Device(monkeypatch)
    inputs = _decode_inputs(64)
    device.home(inputs["state"], nbytes=2 * 64 * _ROW_BYTES)  # a prefix view
    assert whole_call(device, inputs) == _whole_args(inputs, 64 * 32)
    device.home(inputs["state"], device_index=1)
    assert whole_call(device, inputs) == _whole_args(inputs, 64 * 32)
    assert gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES == {}
    device.home(inputs["state"])
    device.table = None
    assert whole_call(device, inputs) == _whole_args(inputs, 64 * 32)
    assert gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES == {}
    device.table = device.make_table((106, 106))
    for batch_size in (16, 32, 47):
        small = _decode_inputs(batch_size)
        device.home(small["state"])
        for _ in range(2):
            assert whole_call(device, small) == _whole_args(small, batch_size * 32)
    launches = gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES
    assert sorted(key[1] for key in launches) == [16, 32, 47]
    assert all(launch[0] is None for launch in launches.values())
    assert not any(_TWIN in name for name in device.loads)
    assert device.counter_calls == []
    for capability, sm_count in (((10, 7), 216), ((10, 0), 148)):
        device = _Device(monkeypatch, capability=capability, sm_count=sm_count)
        inputs = _decode_inputs(64)
        device.home(inputs["state"])
        for _ in range(2):
            args = whole_call(device, inputs)
        assert args[:9] == _whole_args(inputs, 0)[:9]
        assert not any(_TWIN in name for name in device.loads)
        assert device.counter_calls == []


def test_twin_build_failure_is_remembered_and_runs_whole(monkeypatch) -> None:
    device = _Device(monkeypatch)
    device.failing.add(_twin_name(_SCALE))
    inputs = _decode_inputs(64)
    device.home(inputs["state"])
    for _ in range(3):
        _run(inputs)
    assert device.loads.count(_twin_name(_SCALE)) == 1
    assert [name for name, _ in device.calls] == [_whole_name(_SCALE)] * 3
    assert device.counter_calls == []


def test_twin_load_failure_at_its_launch_runs_whole(monkeypatch) -> None:
    # A twin whose kernel fails to load at its first launch (the failure comes before
    # the launch) leaves the call to the row's whole launch, which runs; the twin is
    # dropped with one warning, and every later call on such a state runs the whole
    # launch alone; the whole launch table keeps its value.
    device = _Device(monkeypatch)
    device.failing_launch.add(_twin_name(_SCALE))
    inputs = _decode_inputs(64)
    device.home(inputs["state"])
    for _ in range(3):
        assert _run(inputs) is inputs["output"]
    assert [name for name, _ in device.calls] == [_twin_name(_SCALE)] + [
        _whole_name(_SCALE)
    ] * 3
    assert device.calls[0][1] == _twin_args(inputs, device, 32)
    assert [args for _, args in device.calls[1:]] == [_whole_args(inputs, 64 * 32)] * 3
    assert device.loads.count(_twin_name(_SCALE)) == 1
    assert device.counter_calls == [(locality.DOMAIN_COUNTER_WORDS, 0)]
    key = (0, 64, 16, 16, 32, _SCALE, True)
    assert gdn_decode._CAKE_GDN_NONTRANSPOSE_LAUNCHES[key][1] == 64 * 32
    (launch,) = gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES.items()
    assert launch[0] == key + ((106, 106), 32 * _ROW_BYTES)
    assert launch[1][0] is None and launch[1][1:] == (64 * 32, 32, device.table)
    # The allocation was bound again, to the row's whole-GPU launch.
    ((record, whole, domain, table),) = (
        gdn_decode._CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES.values()
    )
    assert record is device.homings[id(inputs["state"])] and domain is None
    assert table is None
    assert whole is gdn_decode._CAKE_GDN_NONTRANSPOSE_LAUNCHES[key]
    (warning,) = device.warnings
    assert "failed to load or launch" in warning
    assert f"failed to load the kernel of {_twin_name(_SCALE)}" in warning


@pytest.mark.parametrize("launched", [False, True])
def test_twin_launch_error_propagates_and_keeps_the_twin(monkeypatch, launched) -> None:
    # A launch error of the call, not of the twin (the whole launch fails too, as on
    # the legacy default stream while another thread captures on a blocking stream),
    # raises the whole launch's error, as a call on an unhomed state does, and keeps
    # the twin: no warning, the same twin launch and binding, and the next call runs
    # the twin.  That holds at the twin's first launch too.
    device = _Device(monkeypatch)
    inputs = _decode_inputs(64)
    device.home(inputs["state"])
    if launched:
        _run(inputs)
    device.calls.clear()
    device.launch_error = "operation not permitted when stream is capturing"
    with pytest.raises(RuntimeError) as raised:
        _run(inputs)
    assert str(raised.value) == f"{device.launch_error}: {_whole_name(_SCALE)}"
    assert str(raised.value.__context__).endswith(_twin_name(_SCALE))
    assert [name for name, _ in device.calls] == [
        _twin_name(_SCALE),
        _whole_name(_SCALE),
    ]
    assert device.calls[0][1] == _twin_args(inputs, device, 32)
    assert device.calls[1][1] == _whole_args(inputs, 64 * 32)
    assert device.warnings == []
    key = (0, 64, 16, 16, 32, _SCALE, True)
    ((domain_key, launch),) = gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES.items()
    assert domain_key == key + ((106, 106), 32 * _ROW_BYTES)
    assert launch[0] is not None
    record = device.homings[id(inputs["state"])]
    assert gdn_decode._CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES[id(record)][2] is launch
    device.launch_error = None
    device.calls.clear()
    for _ in range(2):
        assert _run(inputs) is inputs["output"]
    assert device.calls == [(_twin_name(_SCALE), _twin_args(inputs, device, 32))] * 2
    assert device.loads.count(_twin_name(_SCALE)) == 1
    assert device.warnings == []


def test_twin_load_failure_is_dropped_where_the_whole_launch_runs(monkeypatch) -> None:
    # A twin that fails to load at a call whose whole launch fails too is kept (the
    # call raises the whole launch's error); the first call whose whole launch runs
    # drops it.
    device = _Device(monkeypatch)
    device.failing_launch.add(_twin_name(_SCALE))
    inputs = _decode_inputs(64)
    device.home(inputs["state"])
    device.launch_error = "unspecified launch failure"
    with pytest.raises(RuntimeError, match="unspecified launch failure"):
        _run(inputs)
    assert device.warnings == []
    (launch,) = gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES.values()
    assert launch[0] is not None
    device.launch_error = None
    for _ in range(2):
        assert _run(inputs) is inputs["output"]
    assert [name for name, _ in device.calls] == [
        _twin_name(_SCALE),
        _whole_name(_SCALE),
        _twin_name(_SCALE),
        _whole_name(_SCALE),
        _whole_name(_SCALE),
    ]
    assert len(device.warnings) == 1
    (launch,) = gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES.values()
    assert launch[0] is None


def test_twin_takes_the_bound_table_and_converted_eager_counters(monkeypatch) -> None:
    # The twin takes the SM table its allocation's binding converted once, on every
    # call, eager or captured; an eager launch takes its stream's converted counter
    # pair, and a captured launch the pair launch_counters returned, unconverted.
    device = _Device(monkeypatch)
    inputs = _decode_inputs(64)
    device.home(inputs["state"])
    for _ in range(2):
        _run(inputs)
    device.capturing = True
    _run(inputs)
    record = device.homings[id(inputs["state"])]
    table = gdn_decode._CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES[id(record)][3]
    assert table == _Converted(device.table.entries)
    eager, again, captured = [args for _, args in device.calls]
    assert eager == again == _twin_args(inputs, device, 32)
    assert captured == _twin_args(inputs, device, 32, captured=True)
    assert all(args[12] is table for args in (eager, again, captured))
    assert eager[10] is again[10] and eager[11] is again[11]
    assert captured[10] is device.counters[0] and captured[11] is device.counters[1]
    assert device.counter_calls == [(locality.DOMAIN_COUNTER_WORDS, 0)] * 2


def test_twin_launch_follows_the_sm_table_and_the_cut(monkeypatch) -> None:
    # A homed allocation keeps the launch it was bound to at its first call, with that
    # call's SM table (a device's table does not change once probed).  A new
    # allocation is bound again through the twin launch table, which reuses a launch
    # only for the SM table object it was resolved with and keys the dies' SM counts
    # and the die cut.
    device = _Device(monkeypatch)
    inputs = _decode_inputs(64)
    device.home(inputs["state"], rows0=32)
    _run(inputs)
    _run(inputs)
    assert device.loads.count(_twin_name(_SCALE)) == 1
    first = device.table
    device.table = device.make_table((106, 106))
    _run(inputs)
    assert device.loads.count(_twin_name(_SCALE)) == 1
    assert device.calls[-1][1][12] == _Converted(first.entries)
    device.home(inputs["state"], rows0=32)
    _run(inputs)
    assert device.loads.count(_twin_name(_SCALE)) == 2
    assert device.calls[-1][1][12] == _Converted(device.table.entries)
    device.home(inputs["state"], rows0=20)
    _run(inputs)
    device.table = device.make_table((100, 112))
    device.home(inputs["state"], rows0=20)
    _run(inputs)
    assert device.calls[-1][1][12] == _Converted(device.table.entries)
    assert [args[9] for _, args in device.calls] == [32, 32, 32, 32, 20, 20]
    keys = sorted(key[-2:] for key in gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES)
    assert keys == [
        ((100, 112), 20 * _ROW_BYTES),
        ((106, 106), 20 * _ROW_BYTES),
        ((106, 106), 32 * _ROW_BYTES),
    ]


def test_twin_launch_table_is_cleared_at_its_limit(monkeypatch) -> None:
    device = _Device(monkeypatch)
    monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_LAUNCH_LIMIT", 3)
    inputs = _decode_inputs(64)
    for rows0 in (1, 2, 3, 4):
        device.home(inputs["state"], rows0=rows0)
        _run(inputs)
    assert len(gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES) == 1
    assert len(gdn_decode._CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES) == 1
    assert [args[9] for _, args in device.calls] == [1, 2, 3, 4]


def test_homed_allocation_is_bound_once(monkeypatch) -> None:
    # The first call on a homed allocation binds it to its row's launch; later calls
    # on it (the same homing record) query no SM table and resolve nothing.  A new
    # allocation and a new whole launch of the row bind again; a binding to the whole
    # launch (a part of an allocation) is kept too; a call that finds no SM table is
    # not bound, so a later call with a table runs the twin.
    device = _Device(monkeypatch)
    binds = []
    bind = gdn_decode._bind_cake_gdn_nontranspose_domain_launch

    def spy(key, whole, state, homed):
        binds.append(homed)
        return bind(key, whole, state, homed)

    monkeypatch.setattr(gdn_decode, "_bind_cake_gdn_nontranspose_domain_launch", spy)
    inputs = _decode_inputs(64)
    device.home(inputs["state"])
    _run(inputs)
    queries = device.table_queries
    for _ in range(3):
        _run(inputs)
    assert len(binds) == 1 and device.table_queries == queries
    assert [name for name, _ in device.calls] == [_twin_name(_SCALE)] * 4
    # A new allocation binds through the twin launch table: no new load.
    device.home(inputs["state"])
    _run(inputs)
    _run(inputs)
    assert len(binds) == 2 and device.loads.count(_twin_name(_SCALE)) == 1
    # A new whole launch of the row (its table was cleared) binds again.
    gdn_decode._CAKE_GDN_NONTRANSPOSE_LAUNCHES.clear()
    _run(inputs)
    _run(inputs)
    assert len(binds) == 3
    assert [name for name, _ in device.calls[4:]] == [_twin_name(_SCALE)] * 4
    # A prefix view of a larger allocation is bound to the whole launch.
    device.calls.clear()
    device.home(inputs["state"], nbytes=2 * 64 * _ROW_BYTES)
    queries = device.table_queries
    for _ in range(2):
        _run(inputs)
    assert len(binds) == 4 and device.table_queries == queries
    assert [name for name, _ in device.calls] == [_whole_name(_SCALE)] * 2
    # No SM table: the whole launch, unbound; then the twin once the table is there.
    device.calls.clear()
    device.home(inputs["state"])
    device.table = None
    _run(inputs)
    device.table = device.make_table((106, 106))
    _run(inputs)
    _run(inputs)
    assert len(binds) == 6
    assert [name for name, _ in device.calls] == [_whole_name(_SCALE)] + [
        _twin_name(_SCALE)
    ] * 2
    assert device.calls[-1][1][12] == _Converted(device.table.entries)


def test_twin_counters_are_reused_per_stream_and_thread(monkeypatch) -> None:
    # Eager twin launches take their stream's pair from launch_counters once and reuse
    # it; another stream or another thread takes its own; each captured launch takes
    # a pair from launch_counters, and the first eager launch after a capture goes
    # through launch_counters again (it tops the capture reserve up).  The pairs kept
    # per thread are dropped past STREAM_PAIR_LIMIT.
    device = _Device(monkeypatch)
    pairs: list[tuple] = []

    def launch_counters(words, device_index=None):
        device.counter_calls.append((words, device_index))
        pairs.append((object(), object()))
        return pairs[-1]

    monkeypatch.setattr(cake_locality, "launch_counters", launch_counters)
    inputs = _decode_inputs(64)
    device.home(inputs["state"])

    def used():
        _run(inputs)
        name, args = device.calls[-1]
        assert name == _twin_name(_SCALE)
        return args[10:12]

    def converted(pair):
        return tuple(_Converted(counter) for counter in pair)

    first = used()
    assert used() == first == converted(pairs[0]) and used() == first
    device.stream = 9
    second = used()
    assert second == converted(pairs[1]) and used() == second
    device.stream = 7
    assert used() == first and len(pairs) == 2
    theirs: list[tuple] = []
    thread = threading.Thread(target=lambda: theirs.append(used()))
    thread.start()
    thread.join()
    assert theirs == [converted(pairs[2])]
    assert used() == first and len(pairs) == 3
    device.capturing = True
    assert [used(), used()] == pairs[3:5]
    device.capturing = False
    after = used()
    assert after == converted(pairs[5]) and used() == after and len(pairs) == 6
    monkeypatch.setattr(cake_locality, "STREAM_PAIR_LIMIT", 2)
    device.stream = 11
    assert used() == converted(pairs[6]) and len(pairs) == 7
    device.stream = 7
    assert used() == converted(pairs[7]) and len(pairs) == 8
    assert device.counter_calls == [(locality.DOMAIN_COUNTER_WORDS, 0)] * 8


# --------------------------------------------------------------------------- helper


class _Helper:
    """Fakes for ``localize_gdn_decode_state`` on cuda:0: refusals before the probe
    raise from the probe and the allocator."""

    def __init__(self, monkeypatch, *, arch="sm_107a", sm_count=212, counts=(106, 106)):
        self.probed = False
        self.allow_probe = False
        self.table = _Device.make_table(counts)
        self.allocations: list[tuple] = []
        self.reserves: list[tuple[int, int]] = []
        self.granule = 2 << 20
        self.capturing = False
        self.reserve_error = False
        monkeypatch.setattr(locality, "_device_info", lambda _i: (arch, sm_count))
        monkeypatch.setattr(
            cake_locality, "_stream_capturing", lambda _i: self.capturing
        )
        monkeypatch.setattr(cake_locality, "sm_table", self.sm_table)
        monkeypatch.setattr(cake_locality, "split_empty", self.split_empty)
        monkeypatch.setattr(cake_locality, "granularity", lambda _d=None: self.granule)
        monkeypatch.setattr(cake_locality, "reserve_launch_counters", self.reserve)

    def sm_table(self, device=None):
        assert self.allow_probe, "probed the SM table for a refused state"
        self.probed = True
        return self.table

    def split_empty(self, shape, dtype, cut, device=None):
        self.allocations.append((tuple(shape), dtype, cut, device))
        return torch.empty(shape, dtype=dtype, device="meta")

    def reserve(self, words, device=None):
        self.reserves.append((words, device))
        if self.reserve_error:
            raise RuntimeError("no memory")


def test_localize_refuses_bad_arguments() -> None:
    with pytest.raises(ValueError, match="CUDA torch.Tensor"):
        locality.localize_gdn_decode_state(torch.zeros(1, 32, 128, 128), 16)
    with pytest.raises(ValueError, match="CUDA torch.Tensor"):
        locality.localize_gdn_decode_state(_fake((32, 128, 128), torch.float32), 16)
    transposed = torch.Tensor._make_subclass(
        _FakeCuda,
        torch.empty((64, 32, 128, 128), device="meta").transpose(2, 3),
    )
    with pytest.raises(ValueError, match="contiguous"):
        locality.localize_gdn_decode_state(transposed, 16)
    state = _fake((64, 32, 128, 128), torch.float32)
    with pytest.raises(ValueError, match="num_q_heads"):
        locality.localize_gdn_decode_state(state, 0)
    for rows0 in (-1, 65):
        with pytest.raises(ValueError, match=r"rows0 must be in \[0, 64\]"):
            locality.localize_gdn_decode_state(state, 16, rows0=rows0)


def test_localize_refuses_before_probing(monkeypatch) -> None:
    # Rows no die class routes (below 48: the small band and 32 to 47 rows), another
    # dtype or head size, a __torch_dispatch__ subclass, capture and another
    # architecture are returned unhomed before the SM table is probed.
    def refused(helper, state, *args, **kwargs):
        assert locality.localize_gdn_decode_state(state, *args, **kwargs) is state
        assert not helper.probed and helper.allocations == [] and helper.reserves == []

    helper = _Helper(monkeypatch)
    refused(helper, _fake((64, 32, 128, 128), torch.float16), 16)
    refused(helper, _fake((64, 32, 64, 64), torch.float32), 16)
    for batch_size in (1, 9, 16, 31, 32, 37, 47):
        refused(helper, _fake((batch_size, 32, 128, 128), torch.float32), 16)
        refused(helper, _fake((batch_size, 32, 128, 128), torch.float32), 16, 1.0)
    state = _fake((64, 32, 128, 128), torch.float32)
    refused(helper, state, 4)
    refused(helper, state, 16, scale=0.5)
    refused(helper, _fake((64, 16, 128, 128), torch.float32), 16)
    refused(helper, _wrapper((64, 32, 128, 128), torch.float32), 16)
    helper.capturing = True
    refused(helper, state, 16)
    for arch, sm_count in (
        ("sm_107a", 216),
        ("sm_100a", 148),
        ("sm_103a", 160),
        (None, 132),
    ):
        refused(_Helper(monkeypatch, arch=arch, sm_count=sm_count), state, 16)


def test_localize_refuses_after_probing(monkeypatch) -> None:
    # No SM table, or a cut off the allocation granularity: the state comes back
    # unhomed and nothing is allocated.
    state = _fake((64, 32, 128, 128), torch.float32)
    helper = _Helper(monkeypatch, counts=None)
    helper.allow_probe = True
    assert locality.localize_gdn_decode_state(state, 16) is state
    assert helper.probed and helper.allocations == []
    helper = _Helper(monkeypatch)
    helper.allow_probe = True
    helper.granule = 4 << 20
    assert locality.localize_gdn_decode_state(state, 16, rows0=13) is state
    assert helper.allocations == [] and helper.reserves == []


@pytest.mark.parametrize("counts", [(106, 106), (100, 112), (112, 100)])
def test_localize_homes_by_the_byte_cut(monkeypatch, counts) -> None:
    # The default cut is half the rows on each die (die 0 the odd row), whatever the
    # dies' SM counts; an explicit rows0 (0 and B included) is the cut; homing fills
    # the capture reserve of one-word counter pairs, and a failing refill does not
    # fail the call.
    helper = _Helper(monkeypatch, counts=counts)
    helper.allow_probe = True
    for batch_size, scale, rows0, expected in (
        (64, None, None, 32),
        (63, None, None, 32),
        (512, 1.0, None, 256),
        (64, _SCALE, 13, 13),
        (64, None, 0, 0),
        (64, None, 64, 64),
    ):
        state = _fake((batch_size, 32, 128, 128), torch.float32)
        homed = locality.localize_gdn_decode_state(state, 16, scale, rows0)
        assert homed is not state
        assert tuple(homed.shape) == tuple(state.shape)
        assert helper.allocations[-1] == (
            tuple(state.shape),
            torch.float32,
            expected * _ROW_BYTES,
            0,
        )
        assert helper.reserves[-1] == (1, 0)
    helper.reserve_error = True
    state = _fake((64, 32, 128, 128), torch.float32)
    assert locality.localize_gdn_decode_state(state, 16) is not state


def test_localize_is_exported_next_to_the_state_pool() -> None:
    assert gdn_decode.localize_gdn_decode_state is locality.localize_gdn_decode_state
    assert callable(gdn_decode.ugpu_state_pool)


def test_twin_launch_args_share_the_memory_of_their_tensors() -> None:
    # The SM table and the eager claim counters reach the twin as tvm-ffi tensors made
    # once; each views the memory of its torch tensor and keeps it alive.
    entries = torch.arange(cake_locality.SM_TABLE_LENGTH, dtype=torch.int32)
    converted = gdn_decode._cake_gdn_launch_arg(entries)
    assert converted is not entries
    assert torch.equal(torch.from_dlpack(converted), entries)
    assert torch.from_dlpack(converted).data_ptr() == entries.data_ptr()
    marker = object()
    assert gdn_decode._cake_gdn_launch_arg(marker) is marker


@pytest.mark.parametrize(
    ("handle", "capturing", "expected", "tested"),
    [
        (0, True, (0, 0), False),
        (1, True, (0, 1), False),
        (2, True, None, True),
        (7, True, None, True),
        (7, False, (0, 7), True),
    ],
)
def test_eager_stream_tests_capture_off_the_legacy_default_stream(
    monkeypatch, handle, capturing, expected, tested
) -> None:
    # The legacy default stream cannot capture a CUDA graph: its eager twin launches
    # skip the capture test.  Any other stream is tested (on the current device by
    # torch's C bindings, on another device by cake_locality._stream_capturing).
    asked = []

    def is_capturing():
        asked.append("current")
        return capturing

    def stream_capturing(_index):
        asked.append("other")
        return capturing

    monkeypatch.setattr(
        torch._C, "_cuda_getCurrentRawStream", lambda _i: handle, raising=False
    )
    monkeypatch.setattr(
        torch._C, "_cuda_isCurrentStreamCapturing", is_capturing, raising=False
    )
    monkeypatch.setattr(cake_locality, "_stream_capturing", stream_capturing)
    monkeypatch.setattr(torch._C, "_cuda_getDevice", lambda: 0, raising=False)
    assert gdn_decode._cake_gdn_eager_stream(0) == expected
    assert asked == (["current"] if tested else [])
    if tested:
        asked.clear()
        monkeypatch.setattr(torch._C, "_cuda_getDevice", lambda: 1, raising=False)
        assert gdn_decode._cake_gdn_eager_stream(0) == expected
        assert asked == ["other"]
