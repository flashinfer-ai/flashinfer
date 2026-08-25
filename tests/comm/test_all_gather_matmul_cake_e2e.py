import random

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp

from flashinfer.comm import all_gather_matmul
from flashinfer.utils import get_compute_capability


def _run_cake_subgroup(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    symm_mem.set_backend("NVSHMEM")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    symm_mem.enable_symm_mem_for_group(group.group_name)
    torch.manual_seed(41 + rank)
    rows = 384
    inp = symm_mem.empty(rows, 8192, dtype=dtype, device=device).normal_()
    weight = torch.randn(8192, 2048, dtype=dtype, device=device)

    for _ in range(2):
        actual = all_gather_matmul(inp, weight, group, backend="cake")
        gathered = torch.empty(
            world_size * rows, 8192, dtype=dtype, device=device
        )
        dist.all_gather_into_tensor(gathered, inp, group=group)
        expected = gathered @ weight
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    dist.destroy_process_group(group)
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() not in (2, 4),
    reason="Cake all-gather matmul e2e requires exactly two or four visible GPUs",
)
@pytest.mark.skipif(
    torch.cuda.device_count() == 0
    or get_compute_capability(torch.device("cuda:0")) not in ((10, 0), (10, 3)),
    reason="Cake all-gather matmul e2e requires SM100 or SM103",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_all_gather_matmul_cake_arbitrary_subgroup(dtype):
    world_size = torch.cuda.device_count()
    port = random.randint(30000, 60000)
    mp.spawn(
        _run_cake_subgroup,
        args=(world_size, port, dtype),
        nprocs=world_size,
        join=True,
    )
