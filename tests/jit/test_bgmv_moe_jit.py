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

import pytest
from packaging.version import Version


@pytest.mark.parametrize(
    ("target_archs", "expected"),
    [
        ({(8, "0")}, False),
        ({(9, "0a")}, True),
        ({(12, "0f")}, True),
    ],
)
def test_detect_bgmv_moe_capability(monkeypatch, target_archs, expected):
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
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version("13.0"))

    assert aot.detect_sm_capabilities()["bgmv_moe"] is expected


@pytest.mark.parametrize("supported", [False, True])
def test_aot_registers_bgmv_moe_only_for_supported_target(monkeypatch, supported):
    from flashinfer import aot

    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn_fmha")
    )
    monkeypatch.setattr(aot, "gen_gemm_module", lambda: SimpleNamespace(name="gemm"))
    monkeypatch.setattr(
        aot, "gen_hash_topk_module", lambda: SimpleNamespace(name="hash_topk")
    )
    monkeypatch.setattr(
        aot, "gen_bgmv_moe_module", lambda: SimpleNamespace(name="bgmv_moe")
    )

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {"bgmv_moe": supported},
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )

    assert ("bgmv_moe" in {spec.name for spec in specs}) is supported
