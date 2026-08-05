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

"""
Shared test setup for vendored Prims-TS kernels.
"""

import importlib.util
from pathlib import Path

import pytest

from flashinfer.prims_ts.cutlass_dsl import (
    get_cutlass_dsl_bootstrap_error,
    require_cutlass_dsl_experimental,
)


require_cutlass_dsl_experimental()


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


HAS_PRIMS_TS_RUNTIME = all(
    _has_module(name)
    for name in (
        "cutlass",
        "cutlass.cute",
        "cutlass.experimental.primitives",
        "cutlass.experimental.task_scheduling",
    )
)


@pytest.fixture(autouse=True)
def _scope_exhaustive_schedule_checker(monkeypatch, request):
    test_path = Path(str(request.node.path))
    if test_path.name.startswith("test_batched_gemm_"):
        monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "1")
    else:
        monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")


def pytest_report_header(config):
    del config
    if HAS_PRIMS_TS_RUNTIME:
        return "Prims-TS runtime dependencies: available via installed CUTLASS DSL wheel"
    error = get_cutlass_dsl_bootstrap_error()
    return (
        "Prims-TS runtime dependencies: unavailable from installed CUTLASS "
        f"DSL wheel; import error: {error!r}"
    )
