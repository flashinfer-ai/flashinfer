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

import pytest

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
