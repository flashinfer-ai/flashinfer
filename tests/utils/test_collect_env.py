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

import json
import sys

import pytest

from flashinfer import collect_env
from flashinfer.collect_env import collect_env_info, format_report


@pytest.fixture(scope="module")
def report():
    # The core contract: collection never raises, regardless of environment
    # (no GPU, missing optional packages, ...). Collect once for all tests.
    return collect_env_info()


def test_sections_present(report):
    for section in (
        "FlashInfer",
        "Python / Platform",
        "GPU / Driver",
        "CUDA Toolkit",
        "PyTorch",
        "GPU Libraries: loaded vs on disk",
        "Relevant Packages",
        "Environment Variables",
    ):
        assert section in report
        assert isinstance(report[section], dict)


def test_report_has_flashinfer_version(report):
    import flashinfer

    assert report["FlashInfer"]["flashinfer"] == flashinfer.__version__


def test_format_report(report):
    text = format_report(report)
    assert "FlashInfer environment report" in text
    assert "==== Relevant Packages ====" in text


def test_json_serializable(report):
    json.dumps(report)


def _kernel_package_info(monkeypatch, dists):
    monkeypatch.setattr(collect_env, "_installed_distributions", lambda: dists)
    return collect_env._get_kernel_package_info()


@pytest.mark.parametrize(
    ("pkg", "version", "matches"),
    [
        ("flashinfer-cubin", "0.7.0", True),
        ("flashinfer-cubin", "0.7.0+cu130", False),
        ("flashinfer-jit-cache", "0.7.0", True),
        ("flashinfer-jit-cache", "0.7.0+cu130", True),
        ("flashinfer-jit-cache", "0.7.0.post1+cu130", False),
        ("flashinfer-jit-cache-sm90a", "0.7.0+cu130", True),
    ],
)
def test_kernel_package_version_rule(pkg, version, matches):
    assert collect_env._kernel_package_matches(pkg, version, "0.7.0") is matches


def test_matching_kernel_packages_not_flagged(monkeypatch):
    info = _kernel_package_info(
        monkeypatch,
        {
            "flashinfer-python": "0.7.0",
            "flashinfer-cubin": "0.7.0",
            "flashinfer-jit-cache": "0.7.0+cu130",
            "flashinfer-jit-cache-sm90a": "0.7.0+cu130",
            "flashinfer-jit-cache-sm100a": "0.7.0+cu130",
        },
    )

    assert info == {
        "flashinfer-cubin": "0.7.0",
        "flashinfer-jit-cache": "0.7.0+cu130",
        "flashinfer-jit-cache providers": "sm100a, sm90a",
    }


def test_stale_kernel_packages_flagged(monkeypatch):
    info = _kernel_package_info(
        monkeypatch,
        {
            "flashinfer-python": "0.7.0.post1",
            "flashinfer-cubin": "0.7.0",
            "flashinfer-jit-cache": "0.7.0+cu130",
            "flashinfer-jit-cache-sm90a": "0.7.0+cu130",
        },
    )

    assert info == {
        "flashinfer-cubin": "0.7.0  ⚠ MISMATCH vs flashinfer-python==0.7.0.post1",
        "flashinfer-jit-cache": (
            "0.7.0+cu130  ⚠ MISMATCH vs flashinfer-python==0.7.0.post1"
        ),
        "flashinfer-jit-cache providers": "sm90a",
    }


def test_jit_cache_providers_must_match_shim(monkeypatch):
    info = _kernel_package_info(
        monkeypatch,
        {
            "flashinfer-python": "0.7.0",
            "flashinfer-jit-cache": "0.7.0+cu130",
            "flashinfer-jit-cache-sm80": "0.7.0+cu129",
            "flashinfer-jit-cache-sm90a": "0.7.0+cu130",
        },
    )

    assert info["flashinfer-jit-cache providers"] == (
        "sm80, sm90a  ⚠ MISMATCH vs flashinfer-jit-cache==0.7.0+cu130: "
        "sm80==0.7.0+cu129"
    )


def test_jit_cache_providers_without_shim(monkeypatch):
    info = _kernel_package_info(
        monkeypatch,
        {"flashinfer-python": "0.7.0", "flashinfer-jit-cache-sm90a": "0.7.0+cu130"},
    )

    assert info["flashinfer-jit-cache"] == "not installed"
    assert info["flashinfer-jit-cache providers"] == (
        "sm90a  ⚠ flashinfer-jit-cache not installed"
    )


def test_kernel_packages_reported_when_import_fails(monkeypatch):
    # A stale kernel wheel can make `import flashinfer` fail; the report must
    # still show the versions that explain it.
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.setattr(
        collect_env,
        "_installed_distributions",
        lambda: {"flashinfer-python": "0.7.0.post1", "flashinfer-cubin": "0.7.0"},
    )

    info = collect_env._get_flashinfer_info()

    assert info["flashinfer"].startswith("<import failed")
    assert info["flashinfer-cubin"] == (
        "0.7.0  ⚠ MISMATCH vs flashinfer-python==0.7.0.post1"
    )
