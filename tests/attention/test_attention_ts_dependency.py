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

from importlib.metadata import PackageNotFoundError

import pytest

from flashinfer.attention._prims_ts_dependency import require_prims_ts_cutlass_dsl


@pytest.mark.parametrize("version", ["4.7.0", "4.7.1", "5.0.0"])
def test_prims_ts_accepts_cutlass_dsl_47_or_newer(version: str) -> None:
    assert require_prims_ts_cutlass_dsl(lambda _: version) == version


@pytest.mark.parametrize("version", ["4.6.0", "4.6.2"])
def test_prims_ts_rejects_cutlass_dsl_46(version: str) -> None:
    with pytest.raises(ImportError, match=r"requires nvidia-cutlass-dsl>=4\.7\.0"):
        require_prims_ts_cutlass_dsl(lambda _: version)


def test_prims_ts_rejects_missing_cutlass_dsl() -> None:
    def missing_distribution(name: str) -> str:
        raise PackageNotFoundError(name)

    with pytest.raises(ImportError, match="but it is not installed"):
        require_prims_ts_cutlass_dsl(missing_distribution)


def test_prims_ts_rejects_invalid_cutlass_dsl_version() -> None:
    with pytest.raises(ImportError, match="invalid version"):
        require_prims_ts_cutlass_dsl(lambda _: "development")
