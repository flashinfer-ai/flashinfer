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

"""Regressions for isolating CUTLASS Task Scheduling import side effects."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


_REPO_ROOT = Path(__file__).parents[1]


def _run_isolated(code: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_REPO_ROOT), env.get("PYTHONPATH")))
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_flashinfer_import_and_availability_probe_do_not_import_task_scheduling():
    _run_isolated(
        """
import sys
import flashinfer

assert "cutlass.experimental.task_scheduling" not in sys.modules
assert "cutlass.experimental.task_scheduling.resources" not in sys.modules

from flashinfer.prims_ts import is_prims_ts_available
assert isinstance(is_prims_ts_available(), bool)
assert "cutlass.experimental.task_scheduling" not in sys.modules
assert "cutlass.experimental.task_scheduling.resources" not in sys.modules

assert callable(flashinfer.prims_ts_bf16_moe)
assert "cutlass.experimental.task_scheduling" not in sys.modules
assert "cutlass.experimental.task_scheduling.resources" not in sys.modules

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType
assert DType.BF16
assert "cutlass.experimental.task_scheduling" not in sys.modules
assert "cutlass.experimental.task_scheduling.resources" not in sys.modules
"""
    )


def test_prims_ts_bootstrap_scopes_work_tile_info_customization():
    from flashinfer.prims_ts import is_prims_ts_available

    if not is_prims_ts_available():
        pytest.skip("CUTLASS DSL is not installed")

    _run_isolated(
        """
from cutlass import Boolean, Int32
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo
from flashinfer.prims_ts.cutlass_dsl import (
    require_cutlass_dsl_experimental,
    task_scheduling_scope,
)

original_init = WorkTileInfo.__init__
require_cutlass_dsl_experimental()
assert WorkTileInfo.__init__ is original_init

flat = WorkTileInfo((Int32(1), Int32(0), Int32(2)), Boolean(True))
nested = WorkTileInfo(
    (Int32(1), Int32(0), (Int32(2), Int32(3))), Boolean(True)
)
assert hasattr(flat, "_tile_idx")
assert hasattr(nested, "_tile_idx")
assert isinstance(nested.tile_idx[2], tuple)
assert len(nested.tile_idx[2]) == 2

with task_scheduling_scope():
    assert WorkTileInfo.__init__ is not original_init
    task_tile = WorkTileInfo((Int32(1), Int32(0), Int32(2)), Boolean(True))
    assert not hasattr(task_tile, "_tile_idx")

assert WorkTileInfo.__init__ is original_init

try:
    with task_scheduling_scope():
        raise RuntimeError("scope restoration probe")
except RuntimeError:
    pass
assert WorkTileInfo.__init__ is original_init

nested_after = WorkTileInfo(
    (Int32(1), Int32(0), (Int32(2), Int32(3))), Boolean(True)
)
assert isinstance(nested_after.tile_idx[2], tuple)

from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    build_batched_gemm_task_manager,
)
assert WorkTileInfo.__init__ is original_init
build_batched_gemm_task_manager(verbose=False)
assert WorkTileInfo.__init__ is original_init
"""
    )


# Emulates a CUTLASS DSL 4.6 wheel: `cutlass.experimental` resolves as an empty
# package so the eager chain inside `import cutlass` is satisfied, while every
# submodule Prims-TS needs (`.primitives`, `.task_scheduling`) stays missing. A
# finder that fails `cutlass.experimental` outright would not simulate 4.6 -- on
# the 4.7 wheel it breaks `import cutlass` itself, which is a different failure.
_DSL46_STUB = """
import importlib.abc
import importlib.machinery
import sys


class _EmptyLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__path__ = []


class _Stub(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cutlass.experimental":
            return importlib.machinery.ModuleSpec(
                fullname, _EmptyLoader(), is_package=True
            )
        return None


sys.meta_path.insert(0, _Stub())
"""


def test_prims_ts_directory_still_collects_without_cutlass_experimental(tmp_path):
    """``tests/prims_ts`` must COLLECT cleanly on a DSL that lacks 4.7.

    ``collect_ignore`` in ``tests/prims_ts/conftest.py`` is hand-maintained: a new
    test module that imports the vendored kernels at module scope cannot be marked
    during collection, only dropped before it, and forgetting to list it silently
    reintroduces #5213 (``ERROR tests/prims_ts``) on every pre-4.7 lane.

    Exit code 0 is the assertion that matters, and it is stricter than it looks:
    a missed module raises during collection (exit 2), and a module-level skip --
    the obvious alternative fix -- collects nothing and exits 5.
    """

    stub = tmp_path / "dsl46_stub.py"
    stub.write_text(_DSL46_STUB)

    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(tmp_path), str(_REPO_ROOT), env.get("PYTHONPATH")))
    )
    env["FLASHINFER_WORKSPACE_BASE"] = str(tmp_path / "workspace")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/prims_ts",
            "--collect-only",
            "-q",
            "--color=no",
            "-p",
            "dsl46_stub",
        ],
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "tests/prims_ts failed to collect under a simulated CUTLASS DSL 4.6.\n"
        "If this is a newly added module that imports the Prims-TS kernels at "
        "module scope, add it to collect_ignore in tests/prims_ts/conftest.py.\n"
        f"exit={result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    assert "test_moe_compile_cache.py" not in result.stdout
    assert (
        "test_moe_api_signature.py::test_prims_ts_fp8_positional_contract_matches_trtllm"
        in result.stdout
    )
