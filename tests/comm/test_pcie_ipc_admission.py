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

Exercise workspace admission exchanges with real CPU Gloo ranks, without CUDA.
"""

from datetime import timedelta

import pytest
import torch.distributed as dist

from flashinfer.comm.pcie_ipc_ar import PcieIpcAllReduceWorkspace
from tests.comm.test_pcie_ipc_all_reduce import multi_process_parallel


def _admission_worker(world_size: int, rank: int, port: int, scenario: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        # Only the admission exchange is under test; CUDA workspace allocation
        # is deliberately not part of this CPU regression.
        workspace = PcieIpcAllReduceWorkspace.__new__(PcieIpcAllReduceWorkspace)
        workspace.group = dist.group.WORLD
        workspace.world_size = world_size

        if scenario == "mixed_capability":
            entries = workspace._joint_check(
                {"error": None, "memop_supported": rank != 0},
                "preparing",
                require_identical=False,
            )
            assert not all(entry["memop_supported"] for entry in entries)
        elif scenario == "preparation_failure":
            local = (
                {"error": "workspace preparation failed"}
                if rank == 0
                else {"error": None, "memop_supported": True}
            )
            with pytest.raises(ValueError, match="failed while preparing"):
                # The failing constructor branch uses the default identical
                # check; successful ranks allow different capabilities.
                entries = workspace._joint_check(
                    local, "preparing", require_identical=rank == 0
                )
                # An error-only peer record must raise jointly before callers
                # access its absent capability field.
                all(entry["memop_supported"] for entry in entries)
        elif scenario == "argument_mismatch":
            with pytest.raises(ValueError, match="identical arguments"):
                workspace._joint_check(
                    {"error": None, "max_numel": 8 * (rank + 1)},
                    "validating arguments",
                )
        else:
            raise AssertionError(f"unknown admission scenario: {scenario}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario", ["mixed_capability", "preparation_failure", "argument_mismatch"]
)
def test_workspace_admission_agrees_across_cpu_ranks(scenario: str) -> None:
    multi_process_parallel(2, _admission_worker, args=(scenario,), timeout_s=60)
