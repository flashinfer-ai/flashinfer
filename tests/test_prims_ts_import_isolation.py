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
