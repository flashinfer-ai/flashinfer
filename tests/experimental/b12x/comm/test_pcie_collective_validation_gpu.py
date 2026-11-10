from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce
from b12x.comm.pcie.pcie_twoshot import PCIeTwoShotSP


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _expect_collective_rejection(operation) -> None:
    with pytest.raises(RuntimeError, match="argument validation"):
        operation()
    # Prove every rank left the same failed setup round and can enter the next.
    dist.barrier()


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    group = dist.group.WORLD
    try:
        _expect_collective_rejection(
            lambda: PCIeOneshotAllReduce.from_exchange_group(
                exchange_group=group,
                device=device,
                eager_buffer_bytes=1 << 20,
                max_size=(2 << 20) if rank == 0 else (1 << 20),
                ext_module=object(),
            )
        )
        _expect_collective_rejection(
            lambda: PCIeDCPA2A.from_exchange_group(
                exchange_group=group,
                device=device,
                max_batch_size=8,
                total_heads=16,
                head_dim=64,
                query_head_dim=7 if rank == 0 else 64,
                ext_module=object(),
            )
        )
        _expect_collective_rejection(
            lambda: PCIeDCPA2A.from_exchange_group(
                exchange_group=group,
                device=torch.device("cpu") if rank == 0 else device,
                max_batch_size=8,
                total_heads=16,
                head_dim=64,
                ext_module=object(),
            )
        )
        _expect_collective_rejection(
            lambda: PCIeTwoShotSP.from_exchange_group(
                exchange_group=group,
                device=device,
                max_rows=8,
                row_elems=17 if rank == 0 else 16,
                ext_module=object(),
            )
        )
        _expect_collective_rejection(
            lambda: PCIeOneshotAllReduce(
                rank=1 if rank == 0 else rank,
                world_size=world_size,
                device=device,
                signal_ptrs=tuple(0 for _ in range(world_size)),
                exchange_group=group,
                ext_module=object(),
            )
        )
    finally:
        dist.destroy_process_group()


def test_asymmetric_argument_failures_reject_all_ranks_before_ipc() -> None:
    if os.getenv("B12X_PCIE_TEST_COLLECTIVE_VALIDATION") != "1":
        pytest.skip("set the collective-validation GPU gate to run this test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip("two CUDA devices are required")
    mp.spawn(
        _worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )
