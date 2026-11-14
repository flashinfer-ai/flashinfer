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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer.jit import fused_moe

GENERIC_MODULE = "fused_moe_90"
H20_MODULE = "fused_moe_90_h20_prebuilt_d"
H100_MODULE = "fused_moe_90_h100_prebuilt_d"
H20_FLAG = "-DFLASHINFER_SM90_MOE_PREBUILT_D_DESCRIPTOR=1"
H100_FLAG = "-DFLASHINFER_SM90_MOE_H100_PREBUILT_D_DESCRIPTOR=1"


def _device(name, sm_count, major=9):
    return SimpleNamespace(name=name, major=major, multi_processor_count=sm_count)


H100_SXM = _device("NVIDIA H100 80GB HBM3", 132)
H20 = _device("NVIDIA H20", 78)


@pytest.fixture
def host(monkeypatch):
    """Patch the device, AOT artifacts and toolchain that module selection probes."""
    state = SimpleNamespace(
        device=H100_SXM,
        aot_modules=set(),
        tools={"/usr/local/cuda/bin/nvcc", "ninja"},
        probes=0,
    )

    def get_device_properties(device):
        state.probes += 1
        if state.device is None:
            raise RuntimeError("no CUDA device")
        return state.device

    def get_aot_artifacts(module_name):
        return ("artifact",) if module_name in state.aot_modules else ()

    def gen_module(nvcc_flags, device_arch, use_fast_build=False, module_name=None):
        # Record the selection instead of generating cutlass sources.
        return SimpleNamespace(
            name=module_name or f"fused_moe_{device_arch}", nvcc_flags=nvcc_flags
        )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(fused_moe.jit_env, "get_aot_artifacts", get_aot_artifacts)
    monkeypatch.setattr(
        fused_moe,
        "shutil",
        SimpleNamespace(which=lambda cmd: cmd if cmd in state.tools else None),
    )
    monkeypatch.setattr(fused_moe, "get_cuda_path", lambda: "/usr/local/cuda")
    monkeypatch.setattr(fused_moe, "is_cuda_version_at_least", lambda version: True)
    monkeypatch.setattr(fused_moe, "gen_cutlass_fused_moe_module", gen_module)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.delenv("FLASHINFER_NVCC", raising=False)
    return state


def _select(**kwargs):
    spec = fused_moe.gen_cutlass_fused_moe_sm90_module(**kwargs)
    return spec.name, {H20_FLAG, H100_FLAG}.intersection(spec.nvcc_flags)


@pytest.mark.cpu_only
def test_measured_h100_selects_prebuilt_d_module(host):
    assert _select() == (H100_MODULE, {H100_FLAG})


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(_device("NVIDIA H100 PCIe", 114), id="h100-pcie"),
        pytest.param(_device("NVIDIA H100 NVL", 132), id="h100-nvl"),
        pytest.param(
            _device("NVIDIA H100 80GB HBM3 MIG 3g.40gb", 60), id="h100-sxm-mig"
        ),
        pytest.param(_device("NVIDIA H100 80GB HBM3", 114), id="h100-name-other-sms"),
        pytest.param(_device("NVIDIA H200", 132), id="h200"),
        pytest.param(_device("NVIDIA H800", 132), id="h800"),
        pytest.param(_device("NVIDIA H100 80GB HBM3", 132, major=10), id="not-sm90"),
        pytest.param(None, id="no-cuda-device"),
    ],
)
def test_other_devices_keep_generic_module(host, device):
    host.device = device
    assert _select() == (GENERIC_MODULE, set())


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("aot_modules", "tools", "nvcc_env", "expected"),
    [
        pytest.param(
            set(), {"/usr/local/cuda/bin/nvcc"}, None, GENERIC_MODULE, id="no-ninja"
        ),
        pytest.param(set(), {"ninja"}, None, GENERIC_MODULE, id="no-nvcc"),
        pytest.param(
            set(),
            {"/opt/nvcc", "ninja"},
            "/opt/nvcc",
            H100_MODULE,
            id="flashinfer-nvcc",
        ),
        pytest.param(
            set(),
            {"/usr/local/cuda/bin/nvcc", "ninja"},
            "/opt/nvcc",
            GENERIC_MODULE,
            id="flashinfer-nvcc-missing",
        ),
        pytest.param({H100_MODULE}, set(), None, H100_MODULE, id="h100-aot-artifact"),
        pytest.param(
            {GENERIC_MODULE, H20_MODULE},
            set(),
            None,
            GENERIC_MODULE,
            id="only-other-aot-artifacts",
        ),
    ],
)
def test_h100_requires_aot_artifact_or_compiler(
    monkeypatch, host, aot_modules, tools, nvcc_env, expected
):
    host.aot_modules = aot_modules
    host.tools = tools
    if nvcc_env is not None:
        monkeypatch.setenv("FLASHINFER_NVCC", nvcc_env)
    assert _select()[0] == expected


@pytest.mark.cpu_only
def test_h100_without_cuda_toolkit_keeps_generic_module(monkeypatch, host):
    def missing_cuda_path():
        raise RuntimeError("Could not find nvcc")

    monkeypatch.setattr(fused_moe, "get_cuda_path", missing_cuda_path)
    assert _select() == (GENERIC_MODULE, set())


@pytest.mark.cpu_only
@pytest.mark.parametrize("device", [H100_SXM, H20], ids=["h100", "h20"])
def test_disable_jit_keeps_generic_module(monkeypatch, host, device):
    host.device = device
    host.aot_modules = {H100_MODULE, H20_MODULE}
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    assert _select() == (GENERIC_MODULE, set())


@pytest.mark.cpu_only
def test_h20_selects_its_prebuilt_d_module(host):
    host.device = H20
    assert _select() == (H20_MODULE, {H20_FLAG})


@pytest.mark.cpu_only
@pytest.mark.parametrize("has_aot", [False, True], ids=["jit", "aot"])
def test_nvcc_override_without_cuda_toolkit(monkeypatch, host, has_aot):
    host.tools = {"/opt/nvcc", "ninja"}
    host.aot_modules = {H100_MODULE} if has_aot else set()
    monkeypatch.setenv("FLASHINFER_NVCC", "/opt/nvcc")
    monkeypatch.setattr(
        fused_moe,
        "get_cuda_path",
        Mock(side_effect=RuntimeError("CUDA toolkit not found")),
    )
    expected = (H100_MODULE, {H100_FLAG}) if has_aot else (GENERIC_MODULE, set())
    assert _select() == expected


@pytest.mark.cpu_only
def test_h20_takes_precedence_over_h100(monkeypatch, host):
    # The two macros are mutually exclusive in the C++ module; never emit both.
    host.device = H20
    h100_gate = Mock(return_value=True)
    monkeypatch.setattr(
        fused_moe, "_should_enable_sm90_moe_h100_prebuilt_d_descriptor", h100_gate
    )
    assert _select() == (H20_MODULE, {H20_FLAG})
    h100_gate.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("device", [H100_SXM, H20], ids=["h100", "h20"])
def test_disabled_prebuilt_d_returns_generic_module(host, device):
    host.device = device
    host.aot_modules = {H100_MODULE, H20_MODULE}
    assert _select(enable_prebuilt_d=False) == (GENERIC_MODULE, set())
    assert host.probes == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("device", [H100_SXM, H20], ids=["h100-host", "h20-host"])
def test_aot_registers_generic_sm90_module_on_any_build_host(monkeypatch, host, device):
    from flashinfer import aot

    host.device = device
    # Exercise registration without generating sources for the other modules.
    for name in dir(aot):
        if name.startswith("gen_") and name != "gen_all_modules":
            spec = SimpleNamespace(name=name)
            result = [spec] if name.endswith("_modules") else spec
            monkeypatch.setattr(aot, name, Mock(return_value=result))
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot,
        "gen_cutlass_fused_moe_sm90_module",
        fused_moe.gen_cutlass_fused_moe_sm90_module,
    )

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {"sm90": True},
        add_comm=False,
        add_gemma=False,
        add_oai_oss=False,
        add_moe=True,
        add_act=False,
        add_misc=False,
        add_xqa=False,
    )

    fused_moe_specs = [spec for spec in specs if spec.name.startswith("fused_moe_90")]
    assert [spec.name for spec in fused_moe_specs] == [GENERIC_MODULE]
    assert not {H20_FLAG, H100_FLAG}.intersection(fused_moe_specs[0].nvcc_flags)
