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

import pytest

from flashinfer.prims_ts.batched_gemm import locality_requirements as requirements


@pytest.mark.cpu_only
@pytest.mark.parametrize("version", [(0, 0), (13, 4)])
def test_locality_register_rejects_old_or_missing_dsl(monkeypatch, version):
    monkeypatch.setattr(requirements, "_dsl_cuda_version", lambda: version)
    assert "CUTLASS DSL built with CUDA 13.5" in (
        requirements.locality_domain_register_unsupported_reason()
    )


@pytest.mark.cpu_only
def test_locality_register_rejects_old_driver(monkeypatch):
    monkeypatch.setattr(requirements, "_dsl_cuda_version", lambda: (13, 5))
    monkeypatch.setattr(requirements, "_driver_version", lambda: 13040)
    assert "cuDriverGetVersion >= 13050" in (
        requirements.locality_domain_register_unsupported_reason()
    )


@pytest.mark.cpu_only
def test_locality_register_rejects_single_domain(monkeypatch):
    monkeypatch.setattr(requirements, "_dsl_cuda_version", lambda: (13, 5))
    monkeypatch.setattr(requirements, "_driver_version", lambda: 13050)
    monkeypatch.setattr(
        "flashinfer.locality_domain.locality_domain_count", lambda _device: 1
    )
    assert "at least two locality domains" in (
        requirements.locality_domain_register_unsupported_reason()
    )


@pytest.mark.cpu_only
def test_locality_register_accepts_supported_environment(monkeypatch):
    monkeypatch.setattr(requirements, "_dsl_cuda_version", lambda: (13, 5))
    monkeypatch.setattr(requirements, "_driver_version", lambda: 13050)
    monkeypatch.setattr(
        "flashinfer.locality_domain.locality_domain_count", lambda _device: 2
    )
    assert requirements.locality_domain_register_unsupported_reason() is None
