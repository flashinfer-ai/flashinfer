"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Host-side contracts of the W8A16 dispatcher's kernel cache, with the compile
mocked out: no GPU, no CuTe-DSL compile."""

import sys

import pytest
import torch

from flashinfer.gemm import w8a16_fp8_dispatch as dispatch

_SM90 = (9, 0)
_SM100 = (10, 0)
_SM103 = (10, 3)


@pytest.fixture
def mocked_build(monkeypatch):
    """Replace the (disk-cached) compile with a recorder; start from an empty
    in-process cache and restore it afterwards."""
    built = []

    def fake_build(kernel_name, compile_fn):
        built.append(kernel_name)
        return object()

    monkeypatch.setattr(dispatch, "_build_kernel", fake_build)
    monkeypatch.setattr(dispatch, "_KERNEL_CACHE", {})
    # The kernel constructors are not run: the compile closure is never called.
    pytest.importorskip("cutlass.cute")
    return built


def test_reduce_kernel_cache_is_per_target(mocked_build):
    """One reduce kernel per compile target: the Blackwell chips share a major."""
    specs, stream = (), None
    a = dispatch._get_reduce_kernel(_SM100, torch.bfloat16, False, 4, specs, stream)
    b = dispatch._get_reduce_kernel(_SM100, torch.bfloat16, False, 4, specs, stream)
    c = dispatch._get_reduce_kernel(_SM103, torch.bfloat16, False, 4, specs, stream)
    d = dispatch._get_reduce_kernel(_SM100, torch.float16, False, 4, specs, stream)
    assert a is b
    assert a is not c and a is not d
    assert len(mocked_build) == 3
    assert mocked_build[0] == mocked_build[1]  # sm100a and sm103a: same name...
    # ...told apart on disk by the module directory (the compile arch) and in
    # the process by the target in the key.
    keys = sorted(k[:2] for k in dispatch._KERNEL_CACHE)
    assert keys == [("reduce", _SM100), ("reduce", _SM100), ("reduce", _SM103)]


def test_sm100_and_sm90_kernel_cache_keys_carry_the_target(mocked_build):
    specs, stream = (), None
    dispatch._get_sm100_kernel(
        _SM100, 2, torch.bfloat16, False, 2, False, 148, specs, 128, 128, 128, stream
    )
    dispatch._get_sm100_kernel(
        _SM103, 2, torch.bfloat16, False, 2, False, 148, specs, 128, 128, 128, stream
    )
    dispatch._get_sm90_kernel(
        _SM90, 5, torch.bfloat16, False, 1, 132, specs, 128, 128, 128, 1, stream
    )
    assert len(mocked_build) == 3
    assert {k[:2] for k in dispatch._KERNEL_CACHE} == {
        ("sm100", _SM100),
        ("sm100", _SM103),
        ("sm90", _SM90),
    }
    # Every codegen parameter is in the on-disk name.
    assert mocked_build[0].startswith("sm100_algo2_bf16_bias0_swz2_rm0_tf128_mac148")
    assert mocked_build[2] == "sm90_algo5_bf16_bias0_grp1_np132_tail0"


def test_launch_runs_under_the_activation_device(monkeypatch):
    """The compile target and the occupancy query read the current device, so a
    call on a non-current GPU is made under its own device context."""
    seen = {}

    class FakeDeviceCtx:
        def __init__(self, device):
            seen["device"] = device

        def __enter__(self):
            seen["entered"] = True

        def __exit__(self, *exc):
            return False

    class FakeAct:
        device = torch.device("cuda", 1)

    def fake_sm90(act, *args):
        seen["inside"] = seen.get("entered", False)

    monkeypatch.setattr(torch.cuda, "device", FakeDeviceCtx)
    monkeypatch.setattr(dispatch, "_target", lambda device: _SM90)
    monkeypatch.setattr(dispatch, "_w8a16_fp8_gemm_sm90", fake_sm90)
    act = FakeAct()
    dispatch.launch_w8a16_fp8_gemm(act, act, act, None, act, 0, None)
    assert seen == {"device": FakeAct.device, "entered": True, "inside": True}


# ---------------------------------------------------------------------------
# The Blackwell scheduler's multiply-and-shift division: the bound under which
# it is used, mirrored in Python
# ---------------------------------------------------------------------------


def _magic(d: int, x_max: int, shift: int) -> int:
    """``_fast_divmod_magic`` of the Blackwell kernel, in Python."""
    magic = ((1 << shift) + d - 1) // d
    return 0 if x_max * d >= (1 << shift) else magic


def _divmod(x: int, d: int, magic: int, shift: int):
    if magic == 0:
        q = x // d
    else:
        q = (x * magic) >> shift
    return q, x - q * d


def test_fast_divmod_exact_inside_its_bound_and_disabled_outside():
    pytest.importorskip("cutlass")
    from flashinfer.gemm.kernels.cute_dsl.w8a16_fp8_gemm_sm100 import (
        _FAST_DIVMOD_SHIFT as shift,
    )

    import random

    rng = random.Random(0)
    for _ in range(20000):
        d = rng.randint(1, 1 << 20)
        x_max = rng.randint(1, (1 << shift) // d)  # x_max * d < 2^shift
        magic = _magic(d, x_max, shift)
        assert magic != 0
        for x in (0, 1, d - 1, d, d + 1, x_max - 1, rng.randint(0, x_max - 1)):
            assert _divmod(x, d, magic, shift) == divmod(x, d), (x, d)
    # The admitted bounds reach past the proven domain (a 1sm128x128 problem with
    # 134403328 weight rows has 1050026 cluster tiles): there the fast path is
    # disabled and the exact division runs.
    d, x_max = 1050026, 2 * 1050026
    assert _magic(d, x_max, shift) == 0
    bad = ((1 << shift) + d - 1) // d
    assert _divmod(d - 1, d, bad, shift) != divmod(d - 1, d)  # the hazard it avoids
    assert _divmod(d - 1, d, 0, shift) == divmod(d - 1, d)


def test_max_active_clusters_is_probed_per_device(monkeypatch):
    """Priming the occupancy answer on device 0 must not serve device 1: the probe
    is constructed for, and cached by, the device asked about."""
    cutlass_utils = pytest.importorskip("cutlass.utils")
    constructed = []

    class FakeInfo:
        def __init__(self, device_id):
            constructed.append(device_id)
            self.device_id = device_id

        def get_max_active_clusters(self, cluster_size):
            return (100 + self.device_id) // cluster_size

    class FakeDeviceCtx:
        def __init__(self, device):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cutlass_utils, "HardwareInfo", FakeInfo)
    monkeypatch.setattr(torch.cuda, "device", FakeDeviceCtx)
    dispatch._max_active_clusters.cache_clear()
    try:
        assert dispatch._max_active_clusters(0, 2) == 50
        assert dispatch._max_active_clusters(0, 2) == 50  # cached
        assert dispatch._max_active_clusters(1, 2) == 50  # (100 + 1) // 2
        assert dispatch._max_active_clusters(1, 1) == 101
        assert constructed == [0, 1, 1]
    finally:
        dispatch._max_active_clusters.cache_clear()


# ---------------------------------------------------------------------------
# The benchmark's CUPTI span attribution, on synthetic records
# ---------------------------------------------------------------------------


def _load_bench():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_mm_bf16_fp8.py"
    spec = importlib.util.spec_from_file_location("bench_mm_bf16_fp8", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def test_bench_iteration_spans_attribute_by_correlation_id():
    """Activities reach an iteration through the launch sharing their correlation id,
    whatever order the records arrive in; one launch may own several activities
    (a memset and a kernel); the span is the earliest start to the latest end; a
    launch outside every iteration's host window is ignored; an iteration with no
    device activity is an error."""
    b = _load_bench()
    launches = [
        b._Launch(1000, 1010, 7),  # iteration 0
        b._Launch(1020, 1030, 8),  # iteration 0
        b._Launch(5000, 5010, 9),  # iteration 1
        b._Launch(9000, 9010, 10),  # no iteration owns it
    ]
    activities = [  # out of order on purpose
        b._DeviceActivity(5100, 5300, 9, "gemm"),
        b._DeviceActivity(1200, 1500, 8, "gemm"),
        b._DeviceActivity(1100, 1150, 7, None),  # the memset of iteration 0
        b._DeviceActivity(1090, 1120, 8, None),
        b._DeviceActivity(9100, 9200, 10, "stray"),
    ]
    spans = b._iteration_spans(launches, activities, [(1000, 1030), (4990, 5010)])
    assert (spans[0].start, spans[0].end) == (1090, 1500)
    assert sorted(a.correlation_id for a in spans[0].activities) == [7, 8, 8]
    assert (spans[1].start, spans[1].end) == (5100, 5300)
    # Inclusive host boundaries: a launch starting exactly at the window's end counts.
    spans = b._iteration_spans(launches, activities, [(1020, 1020)])
    assert sorted(a.correlation_id for a in spans[0].activities) == [8, 8]
    with pytest.raises(ValueError, match="iteration 0"):
        b._iteration_spans(launches, activities, [(2000, 3000)])
