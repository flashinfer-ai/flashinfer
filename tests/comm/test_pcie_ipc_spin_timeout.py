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

FLASHINFER_PCIE_IPC_TIMEOUT_S: a peer that never arrives must fail the call
instead of leaving the other ranks spinning forever.
"""

import os
import time

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm.pcie_ipc_ar import _spin_timeout_ns
from flashinfer.jit.comm import gen_pcie_ipc_comm_module
from tests.comm.test_pcie_ipc_all_reduce import (
    _init_process_group,
    multi_process_parallel,
)

_TIMEOUT_S = 3


def test_timeout_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLASHINFER_PCIE_IPC_TIMEOUT_S", raising=False)
    assert _spin_timeout_ns() == 0
    monkeypatch.setenv("FLASHINFER_PCIE_IPC_TIMEOUT_S", "0")
    assert _spin_timeout_ns() == 0
    monkeypatch.setenv("FLASHINFER_PCIE_IPC_TIMEOUT_S", "2.5")
    assert _spin_timeout_ns() == 2_500_000_000
    monkeypatch.setenv("FLASHINFER_PCIE_IPC_TIMEOUT_S", "soon")
    with pytest.raises(ValueError, match="FLASHINFER_PCIE_IPC_TIMEOUT_S"):
        _spin_timeout_ns()


def test_default_module_is_unchanged() -> None:
    # The default (and AOT) build must keep its name and flags: the guard is
    # compiled out unless a timeout is requested.
    spec = gen_pcie_ipc_comm_module()
    assert spec.name == "pcie_ipc_comm"
    flags = getattr(spec, "extra_cuda_cflags", None) or []
    assert not any("SPIN_TIMEOUT" in f for f in flags)


def test_timeout_module_is_separate() -> None:
    spec = gen_pcie_ipc_comm_module(5_000_000_000)
    assert spec.name != "pcie_ipc_comm"
    flags = getattr(spec, "extra_cuda_cflags", None) or []
    assert "-DFLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS=5000000000ULL" in flags


def _missing_peer_worker(world_size: int, rank: int, port: int) -> None:
    # Read once, at module build: must be set before the first workspace.
    os.environ["FLASHINFER_PCIE_IPC_TIMEOUT_S"] = str(_TIMEOUT_S)
    import flashinfer.comm as comm

    _init_process_group(world_size, rank, port)
    group = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    ws = comm.PcieIpcAllReduceWorkspace(
        group=group, max_numel=4096, dtype=torch.bfloat16
    )
    inp = torch.ones(1, 4096, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        ws.all_reduce(inp)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    if rank != 0:
        # Never enter the next call: rank 0 now waits for a peer that is gone.
        time.sleep(10 * _TIMEOUT_S)
        os._exit(0)

    start = time.monotonic()
    try:
        ws.all_reduce(inp)
        torch.cuda.synchronize()
    except RuntimeError:
        # The trap poisons the CUDA context, so no teardown: exit directly.
        os._exit(0 if time.monotonic() - start < 5 * _TIMEOUT_S else 2)
    os._exit(1)  # returned although its peer never arrived


@pytest.mark.node_exclusive
def test_missing_peer_traps_instead_of_hanging() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs with CUDA IPC")
    multi_process_parallel(2, _missing_peer_worker, timeout_s=20 * _TIMEOUT_S)
