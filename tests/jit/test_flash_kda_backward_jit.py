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

import re
from types import SimpleNamespace

import pytest
from packaging.version import Version

from flashinfer.jit import core as jit_core
from flashinfer.jit import flash_kda_backward


@pytest.mark.parametrize(
    ("target", "arch_flag"),
    [
        ("sm100a", "-gencode=arch=compute_100a,code=sm_100a"),
        ("sm103a", "-gencode=arch=compute_103a,code=sm_103a"),
    ],
)
def test_flash_kda_backward_jit_spec_is_exact_blackwell(monkeypatch, target, arch_flag):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "3a")},
    )
    flash_kda_backward.gen_flash_kda_backward_module.cache_clear()

    uri = flash_kda_backward.get_flash_kda_backward_uri(target)
    spec = flash_kda_backward.gen_flash_kda_backward_module(target)

    assert re.fullmatch(rf"flash_kda_backward_[0-9a-f]{{10}}_{target}", uri)
    assert spec.name == uri
    assert spec.sources == [
        flash_kda_backward._get_flash_kda_backward_csrc_dir()
        / "flashkda_backward_binding.cu",
        flash_kda_backward._get_flash_kda_backward_csrc_dir()
        / "flashkda_backward_v483_binding.cu",
    ]
    assert all(source.is_file() for source in spec.sources)
    assert arch_flag in spec.extra_cuda_cflags
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1

    generated_binding = spec.sources[0].read_text()
    assert '#include "flashkda_backward.cu"' in generated_binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_low" in generated_binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_high" in generated_binding
    assert (
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_c16_backward" in spec.sources[1].read_text()
    )
    flash_kda_backward.gen_flash_kda_backward_module.cache_clear()


def test_flash_kda_backward_getter_uses_one_module(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        flash_kda_backward,
        "load_flash_kda_backward_module",
        lambda target: (target, sentinel),
    )
    assert flash_kda_backward.get_flash_kda_backward_module("sm100a") == (
        "sm100a",
        sentinel,
    )


def test_flash_kda_backward_binding_contract():
    binding = (
        flash_kda_backward._get_flash_kda_backward_csrc_dir()
        / "flashkda_backward_binding.cu"
    ).read_text()

    assert "CheckExactBlackwellTarget" in binding
    assert "FLASHINFER_FLASH_KDA_TARGET_MINOR" in binding
    assert "scale=1/sqrt(128)" in binding
    assert "lower_bound=-5.0" in binding
    assert "descriptor_storage must provide at least 768 bytes" in binding
    assert "descriptor_storage must be 64-byte aligned" in binding
    assert "reinterpret_cast<cudaStream_t>" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_low" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_high" in binding


@pytest.mark.parametrize(
    ("target_archs", "cuda_version", "expected"),
    [
        ({(10, "3a")}, "12.9", True),
        ({(10, "3a")}, "12.8", False),
        ({(10, "0a")}, "12.9", False),
    ],
)
def test_flash_kda_backward_aot_capability_gate(
    monkeypatch, target_archs, cuda_version, expected
):
    from flashinfer import aot

    class FakeCompilationContext:
        TARGET_CUDA_ARCHS = target_archs

        def get_nvcc_flags_list(self, supported_major_versions=None):
            del supported_major_versions
            return [
                f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
                for major, minor in sorted(self.TARGET_CUDA_ARCHS)
            ]

    monkeypatch.setattr(aot, "CompilationContext", FakeCompilationContext)
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version(cuda_version))
    capabilities = aot.detect_sm_capabilities()
    assert capabilities["flash_kda_backward_sm103a"] is expected


@pytest.mark.parametrize(
    ("target_archs", "cuda_version", "expected"),
    [
        ({(10, "0a")}, "12.8", True),
        ({(10, "0a")}, "12.7", False),
        ({(10, "3a")}, "12.9", False),
    ],
)
def test_flash_kda_backward_sm100a_aot_capability_gate(
    monkeypatch, target_archs, cuda_version, expected
):
    from flashinfer import aot

    class FakeCompilationContext:
        TARGET_CUDA_ARCHS = target_archs

        def get_nvcc_flags_list(self, supported_major_versions=None):
            del supported_major_versions
            return []

    monkeypatch.setattr(aot, "CompilationContext", FakeCompilationContext)
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version(cuda_version))
    capabilities = aot.detect_sm_capabilities()
    assert capabilities["flash_kda_backward_sm100a"] is expected


def test_aot_registers_flash_kda_backward(monkeypatch):
    from flashinfer import aot

    calls = []

    def fake_flash_kda_backward(target):
        calls.append(target)
        return SimpleNamespace(name=f"flash_kda_backward_{target}")

    training_calls = []

    def fake_flash_kda_training(target):
        training_calls.append(target)
        return SimpleNamespace(name=f"flash_kda_training_{target}")

    monkeypatch.setattr(
        aot,
        "gen_flash_kda_backward_module",
        fake_flash_kda_backward,
    )
    monkeypatch.setattr(
        aot,
        "gen_flash_kda_training_module",
        fake_flash_kda_training,
    )
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {
            "flash_kda_backward_sm100a": True,
            "flash_kda_backward_sm103a": True,
        },
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    assert calls == ["sm100a", "sm103a"]
    assert training_calls == ["sm100a", "sm103a"]
    assert [spec.name for spec in specs] == [
        "spdlog",
        "flash_kda_backward_sm100a",
        "flash_kda_training_sm100a",
        "flash_kda_backward_sm103a",
        "flash_kda_training_sm103a",
        "cudnn",
    ]
