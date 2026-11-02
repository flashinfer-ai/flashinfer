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
