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

import os
import zipfile
from importlib import resources
from pathlib import Path

import pytest


_PACKAGE_NAME = "flashinfer.prims_ts.moe"
_PACKAGE_PATH = "flashinfer/prims_ts/moe"
_CONFIG_NAME = "prims_ts_moe_configs.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prims_ts_moe_config_is_declared_as_package_data():
    pyproject = _PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is only available in source-tree test runs")

    contents = pyproject.read_text(encoding="utf-8")
    assert f'"{_PACKAGE_NAME}" = ["{_CONFIG_NAME}"]' in contents


def test_prims_ts_moe_config_is_available_as_runtime_resource():
    config = resources.files(_PACKAGE_NAME).joinpath(_CONFIG_NAME)
    assert config.is_file()
    assert config.read_text(encoding="utf-8").strip()


def test_prims_ts_moe_config_is_in_prebuilt_wheel():
    wheel_env = os.environ.get("FLASHINFER_TEST_WHEEL")
    if wheel_env is None:
        pytest.skip("set FLASHINFER_TEST_WHEEL to inspect a prebuilt wheel")

    wheel = Path(wheel_env).expanduser()
    assert wheel.is_file(), f"FLASHINFER_TEST_WHEEL is not a file: {wheel}"
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    assert f"{_PACKAGE_PATH}/{_CONFIG_NAME}" in members
