"""Qualify PCIe pool rollback and close with an unrelated current CUDA device."""

from __future__ import annotations

import os
import socket
import time
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import b12x
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_TEARDOWN_TEST") != "1",
    reason="set B12X_RUN_PCIE_TEARDOWN_TEST=1 for two-GPU PCIe teardown tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _check_reduction(output: torch.Tensor, expected: float) -> None:
    torch.testing.assert_close(
        output, torch.full_like(output, expected), rtol=0, atol=0
    )


def _worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    other_device = 1 - rank
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    group = dist.group.WORLD
    print(
        f"rank={rank} pool_device={device} "
        f"gpu_uuid={torch.cuda.get_device_properties(device).uuid}",
        flush=True,
    )
    pool = None
    try:
        for cycle in range(2):
            torch.cuda.set_device(rank)
            pool = PCIeOneshotAllReducePool.from_process_group(
                process_group=group, device=device, max_input_bytes=1 << 16
            )
            pool.prepare_channels(("eager:retained",))
            inp = torch.full((4, 1024), rank + 1, device=device, dtype=torch.bfloat16)
            output = torch.empty_like(inp)
            retained = pool.for_stream(channel_id="eager:retained")
            pool.all_reduce(inp, out=output, channel_id="eager:retained")
            torch.cuda.synchronize(device)
            _check_reduction(output, 3)
            checkpoint = pool.checkpoint_channels()
            pool.prepare_channels(("graph:transient",))
            transient = pool._logical_channels["graph:transient"]
            assert transient._owned_buffers and retained._owned_buffers

            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), pool.capture(
                stream=stream, channel_id="graph:transient"
            ):
                for _ in range(2):
                    pool.all_reduce(inp, out=output)
                stream.synchronize()
                dist.barrier(group=group, device_ids=[rank])
                with kernel_resolution_guard('PCIe teardown graph preparation'):
                    with torch.cuda.graph(graph, stream=stream):
                        pool.all_reduce(inp, out=output)
            torch.cuda.synchronize(device)
            addresses = (inp.data_ptr(), output.data_ptr())
            for step in range(3):
                inp.fill_(rank + 1 + step)
                output.fill_(float("nan"))
                dist.barrier(group=group, device_ids=[rank])
                allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                allocated_bytes = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
                assert torch.cuda.memory_allocated(device) == allocated_bytes
                assert (inp.data_ptr(), output.data_ptr()) == addresses
                _check_reduction(output, 3 + 2 * step)
            graph.reset()
            torch.cuda.synchronize(device)

            barriers = []
            real_barrier = dist.barrier

            def record_barrier(*args, **kwargs):
                barriers.append(kwargs.get("device_ids"))
                return real_barrier(*args, **kwargs)

            torch.cuda.set_device(other_device)
            with patch.object(dist, "barrier", record_barrier):
                pool.rollback_channels(checkpoint)
            assert torch.cuda.current_device() == other_device
            assert barriers == [[rank]] * 3
            assert transient._ipc_imports_closed and transient._ipc_exports_freed
            assert not transient._owned_buffers
            assert not retained._ipc_imports_closed and retained._owned_buffers
            assert pool.checkpoint_channels() == checkpoint

            torch.cuda.set_device(rank)
            inp.fill_(rank + 1)
            pool.all_reduce(inp, out=output, channel_id="eager:retained")
            torch.cuda.synchronize(device)
            _check_reduction(output, 3)
            barriers.clear()
            torch.cuda.set_device(other_device)
            with patch.object(dist, "barrier", record_barrier):
                pool.close()
                pool.close()
            assert torch.cuda.current_device() == other_device
            assert barriers == [[rank]] * 3
            assert retained._ipc_imports_closed and retained._ipc_exports_freed
            assert not retained._owned_buffers and not pool._all_channels
            assert pool._closed
            print(f"rank={rank} cycle={cycle} graph/rollback/close passed", flush=True)
            pool = None
    finally:
        torch.cuda.set_device(rank)
        if pool is not None and not pool._closed:
            pool.close()
        dist.destroy_process_group()


def test_pool_teardown_uses_resident_device_after_graph_replay() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two assigned CUDA devices are required")
    context = mp.spawn(_worker, args=(_free_port(),), nprocs=2, join=False)
    deadline = time.monotonic() + 180
    while not context.join(timeout=1):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
            pytest.fail("PCIe teardown workers exceeded the 180-second deadline")
