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

from pathlib import Path

import pytest

from flashinfer.prims_ts.cutlass_dsl import (
    ensure_cutlass_dsl_experimental,
    get_cutlass_dsl_bootstrap_error,
)

# Prims-TS needs ``cutlass.experimental.primitives`` and
# ``cutlass.experimental.task_scheduling``, which exist only in CUTLASS DSL 4.7+.
# FlashInfer's declared floor is deliberately lower so it stays co-installable
# with vLLM and SGLang, so an environment that cannot provide them is supported
# and must degrade rather than fail -- the same contract the other 4.7-gated
# features honor (``flashinfer/attention/cute_dsl/sm120_fmha.py``,
# ``flashinfer/kda.py``). ``ensure_...`` records the import failure for
# ``pytest_report_header`` instead of raising.
HAS_PRIMS_TS_RUNTIME = ensure_cutlass_dsl_experimental()

# Every module here except this one imports without the wheel; this one reaches
# the vendored kernels at import time, so it cannot be collected at all and has
# to be dropped before collection rather than skipped during it.
if not HAS_PRIMS_TS_RUNTIME:
    collect_ignore = ["test_batched_gemm_captured_schedule_tasks.py"]


def pytest_collection_modifyitems(config, items):
    """Skip this directory when the installed DSL cannot provide Prims-TS.

    Marking collected items rather than skipping at module level is deliberate:
    a module-level skip collects nothing, and a run scoped to this directory
    would then exit with pytest's "no tests collected" code 5 -- trading a red
    for a red. Marked items are collected first, so the run reports skips and
    exits 0.
    """

    del config
    if HAS_PRIMS_TS_RUNTIME:
        return

    here = Path(__file__).parent
    skip_prims_ts = pytest.mark.skip(
        reason=(
            "Prims-TS needs cutlass.experimental from CUTLASS DSL 4.7+; the "
            f"installed wheel cannot provide it: {get_cutlass_dsl_bootstrap_error()!r}"
        )
    )
    for item in items:
        if Path(str(item.path)).parent == here:
            item.add_marker(skip_prims_ts)


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
        return (
            "Prims-TS runtime dependencies: available via installed CUTLASS DSL wheel"
        )
    error = get_cutlass_dsl_bootstrap_error()
    return (
        "Prims-TS runtime dependencies: unavailable from installed CUTLASS "
        f"DSL wheel; import error: {error!r}"
    )
