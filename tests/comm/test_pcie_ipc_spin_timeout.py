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

import multiprocessing as mp
import os
import socket
import time
from typing import Any

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.node_exclusive

# Enough iterations that the dying rank is mid-loop, not mid-setup; few enough
# that the whole case fits in a bounded CI window even with the deadline.
DEAD_AFTER = 8
ITERATIONS = 200


def get_open_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_ranks(
    world_size: int, target: Any, args: tuple = (), timeout_s: float = 120.0
):
    """Run ``target`` on every rank with a bounded wait, tolerating one death.

    Unlike the correctness harness, a rank exiting nonzero is expected here:
    the whole point of the spin deadline is that a survivor of a dead peer
    *errors out* instead of hanging. What must never happen is a process that
    is still alive at the end - that is the hang the deadline exists to remove.
    """
    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=target, args=(world_size, rank, port) + args, name=f"Worker-{rank}"
        )
        p.start()
        procs.append(p)
    deadline = time.monotonic() + timeout_s
    outcomes = []
    try:
        for rank, p in enumerate(procs):
            p.join(timeout=max(1.0, deadline - time.monotonic()))
            assert not p.is_alive(), (
                f"rank {rank} is still alive after {timeout_s}s: "
                "the spin deadline did not fire"
            )
            outcomes.append(p.exitcode)
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
        for p in procs:
            p.join(timeout=30.0)
    return outcomes


def _make_workspace():
    import flashinfer.comm as comm

    ws = comm.PcieIpcAllReduceWorkspace(
        group=dist.group.WORLD, max_numel=2048, dtype=torch.bfloat16, max_blocks=128
    )
    return ws


def _peer_death_worker(world_size: int, rank: int, port: int, use_graph: bool):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size
    )
    ws = _make_workspace()
    want = float(world_size)
    for i in range(ITERATIONS):
        if rank == 1 and i >= DEAD_AFTER:
            # Die where a production peer dies: mid-loop, holding no lock the
            # survivor could observe. SIGKILL so nothing can run on the way
            # down - no IPC unmap, no flag store.
            os.kill(os.getpid(), 9)
        x = torch.full((2048,), float(i + 1), dtype=torch.bfloat16, device="cuda")
        y = ws.all_reduce(x)
        torch.cuda.synchronize()
        assert abs(y.float().mean().item() - (i + 1) * want) < 0.5, f"iter {i}"
    dist.destroy_process_group()


def _graph_replay_worker(
    world_size: int, rank: int, port: int, replay_after_death: int
):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size
    )
    ws = _make_workspace()
    x = torch.full((2048,), 3.0, dtype=torch.bfloat16, device="cuda")

    # Capture one call. Replays must be ordered against every other use of the
    # workspace (see bind_stream); this worker uses nothing else, so a plain
    # replay loop is that ordering.
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        ws.all_reduce(x)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        y = ws.all_reduce(x)

    for i in range(ITERATIONS):
        if rank == 1 and i >= DEAD_AFTER:
            os.kill(os.getpid(), 9)
        g.replay()
        torch.cuda.synchronize()
        # The replay's output tensor stays alive here; reading it also proves
        # the replay really ran, not just returned.
        assert abs(y.float().mean().item() - 3.0 * world_size) < 0.5, f"iter {i}"
    dist.destroy_process_group()


def _correctness_worker(world_size: int, rank: int, port: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size
    )
    ws = _make_workspace()
    for i in range(ITERATIONS):
        x = torch.full((2048,), float(i + 1), dtype=torch.bfloat16, device="cuda")
        y = ws.all_reduce(x)
        torch.cuda.synchronize()
        want = float(i + 1) * world_size
        got = y.float().mean().item()
        assert abs(got - want) < 0.5, f"iter {i}: got {got}, want {want}"
    dist.destroy_process_group()


@pytest.mark.parametrize("use_graph", [False, True])
def test_spin_timeout_turns_peer_death_into_error(use_graph: bool):
    """A SIGKILLed peer must cost the survivor an error, never a hang.

    With the deadline compiled in, the survivor's next collective - eager or a
    captured graph replay, the shape from #9108 that no host-side watchdog can
    interrupt - traps within the deadline and the process exits nonzero. The
    pre-fix behavior for both is an unbounded spin.
    """
    os.environ["FLASHINFER_PCIE_IPC_TIMEOUT_S"] = "8"
    try:
        target = _graph_replay_worker if use_graph else _peer_death_worker
        outcomes = _run_ranks(2, target, (use_graph,) if not use_graph else (True,))
    finally:
        os.environ.pop("FLASHINFER_PCIE_IPC_TIMEOUT_S", None)
    # The dead peer: SIGKILL. The survivor: a CUDA error surfacing as a
    # nonzero exit. Nobody alive, nobody hanging.
    assert outcomes[1] == -9, f"the dying rank should be SIGKILLed, got {outcomes[1]}"
    assert outcomes[0] != 0, (
        "the survivor exited cleanly; peer death should surface as a CUDA error "
        "after the deadline, not be swallowed"
    )


def test_spin_timeout_module_stays_correct():
    """The deadline compiled in must not change a healthy run's results."""
    os.environ["FLASHINFER_PCIE_IPC_TIMEOUT_S"] = "8"
    try:
        outcomes = _run_ranks(2, _correctness_worker)
    finally:
        os.environ.pop("FLASHINFER_PCIE_IPC_TIMEOUT_S", None)
    # A healthy run must exit cleanly on every rank: only the peer-death tests
    # accept nonzero exits.
    assert outcomes == [0, 0], outcomes


def test_spin_timeout_env_plumbing():
    """Cache-keying: unset shares the default module; a value splits it."""
    from flashinfer.jit.comm import _pcie_ipc_spin_cflags

    os.environ.pop("FLASHINFER_PCIE_IPC_TIMEOUT_S", None)
    assert _pcie_ipc_spin_cflags("pcie_ipc_comm") == ("pcie_ipc_comm", [])
    os.environ["FLASHINFER_PCIE_IPC_TIMEOUT_S"] = "5"
    try:
        name, flags = _pcie_ipc_spin_cflags("pcie_ipc_comm")
        assert name == "pcie_ipc_comm_spin5000000000"
        assert flags == ["-DFLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS=5000000000"]
        os.environ["FLASHINFER_PCIE_IPC_TIMEOUT_S"] = "0"
        assert _pcie_ipc_spin_cflags("pcie_ipc_comm") == ("pcie_ipc_comm", [])
    finally:
        os.environ.pop("FLASHINFER_PCIE_IPC_TIMEOUT_S", None)
    with pytest.raises(ValueError):
        os.environ["FLASHINFER_PCIE_IPC_TIMEOUT_S"] = "-1"
        try:
            _pcie_ipc_spin_cflags("pcie_ipc_comm")
        finally:
            os.environ.pop("FLASHINFER_PCIE_IPC_TIMEOUT_S", None)
