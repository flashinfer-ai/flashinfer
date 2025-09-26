import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import pynvml

from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlConfig, MnnvlMemory
from flashinfer.comm.mnnvl import CommBackend as CommBackend


pynvml.nvmlInit()


class CustomCommunicator(CommBackend):
    def __init__(self, group):
        self._group = group

    def Get_rank(self) -> int:
        return dist.get_rank(self._group)

    def Get_size(self) -> int:
        return dist.get_world_size(self._group)

    def allgather(self, data: int | bytes):
        device = f"cuda:{torch.cuda.current_device()}"
        if isinstance(data, int):
            local_tensor = torch.tensor([data], device=device, dtype=torch.int32)
            world_size = self.Get_size()
            gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]

            dist.all_gather(gathered, local_tensor, group=self._group)
            return [int(x.item()) for x in gathered]

        elif isinstance(data, bytes):
            local_tensor = torch.ByteTensor(list(data)).unsqueeze(0).to(device)
            world_size = self.Get_size()
            gathered = [data] * self.Get_size()
            dist.all_gather_object(gathered, data, group=self._group)
            return gathered
        else:
            raise TypeError(f"Unsupported type for allgather: {type(data)}")

    def Split(self, color: int, key: int) -> "CustomCommunicator":
        return self


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int, dtype: torch.dtype, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, dtype, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


def align_memory(size: int):
    align_size = 2 * 1024 * 1024
    return (size + align_size - 1) // align_size * align_size


def _init_mnnvl_memory(world_size, rank, dtype, distributed_init_port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    torch.cuda.set_device(rank)
    MnnvlMemory.initialize()
    mapping = Mapping(world_size, rank, world_size, tp_size=world_size)

    allocate0_size = 4 * 1024 * 1024 - 3 * 1024
    mnnvl_config = MnnvlConfig(
        comm_backend=CustomCommunicator(group),
        fabric_page_size=1 << 29,  # 512MB
        allocation_granularity=0,  # Auto-detect
    )
    MnnvlMemory.set_comm_from_config(mapping, mnnvl_config)
    mnnvl_memory0 = MnnvlMemory(mapping, allocate0_size)
    allocate0_size_aligned = align_memory(allocate0_size)

    assert MnnvlMemory.current_mem_offset == allocate0_size_aligned
    tensor0 = mnnvl_memory0.as_torch_strided_tensor(torch.int32)
    numel_per_rank = allocate0_size // 4
    tensor0[(rank + 1) % world_size] = torch.arange(
        start=rank, end=rank + numel_per_rank, device="cuda"
    )
    dist.barrier(group=group)
    for r in range(world_size):
        torch.equal(
            tensor0[(r + 1) % world_size],
            torch.arange(start=r, end=r + numel_per_rank, device="cuda"),
        )

    allocate1_size = 30 * 1024 * 1024 - 2 * 1024
    mnnvl_memory1 = MnnvlMemory(mapping, allocate1_size)
    allocate1_size_aligned = align_memory(allocate1_size)
    assert (
        MnnvlMemory.current_mem_offset
        == allocate0_size_aligned + allocate1_size_aligned
    )
    tensor1 = mnnvl_memory1.as_torch_strided_tensor(torch.float32)
    numel_per_rank = allocate1_size // 4
    tensor1[(rank + 5) % world_size] = torch.arange(
        start=rank,
        end=rank + numel_per_rank,
        dtype=torch.float32,
        device="cuda",
    )
    dist.barrier(group=group)
    for r in range(world_size):
        torch.equal(
            tensor1[(r + 5) % world_size],
            torch.arange(
                start=r, end=r + numel_per_rank, dtype=torch.float32, device="cuda"
            ),
        )
    dist.barrier(group=group)
    del tensor0, mnnvl_memory0
    dist.barrier(group=group)

    large_allocation2_size = 768 * 1024 * 1024
    large_mnnvl_memory2 = MnnvlMemory(mapping, large_allocation2_size)
    allocate2_size_aligned = align_memory(large_allocation2_size)
    assert MnnvlMemory.current_mem_offset == allocate2_size_aligned
    assert large_mnnvl_memory2.rank_stride == (1 << 30)

    del tensor1


@pytest.mark.skipif(
    not MnnvlMemory.supports_mnnvl(),
    reason="Mnnvl memory is not supported on this platform",
)
@pytest.mark.parametrize("world_size", [2, 4])
def test_mnnvl_custom_communicator(world_size):
    dtype = torch.float16
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )

    print(f"Running test for world_size={world_size}")

    multi_process_parallel(
        world_size,
        dtype,
        _init_mnnvl_memory,
        target_args=(),
    )
    print(f"custom mnnvl communicator world_size = {world_size}: OK")
