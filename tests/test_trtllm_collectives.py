# import multiprocessing as mp
# import socket
from typing import Any

import pytest
import torch

import flashinfer.comm as comm
from flashinfer import MoE_Mapping

# import torch.distributed as dist


RANDOM_SEED = 42
NUM_REPEATS = 8 # To test input as a list 


def _run_reduce_scatter_worker(rank, world_size, dtype):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    mapp = MoE_Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
    )
    hidden_dim = 32

    sizes = [world_size * (i + 1) for i in range(world_size)]
    total_size = sum(sizes)
    shape = (world_size, total_size, hidden_dim)

    input_tensors = [
        torch.randn(shape, dtype=dtype, device=device) for _ in range(NUM_REPEATS)
    ]
    expected_output = [i.sum(dim=0) for i in input_tensors]
    input_rs = [i[rank,:,:] for i in input_tensors]

    output = comm.reduce_scatter(
        input_rs,
        mapp,
        dim=0,
        sizes=sizes,
    )

    for i in range(NUM_REPEATS):
        start = sum(sizes[:rank])
        end = start + sizes[rank]
        torch.testing.assert_close(
            output[i], expected_output[i][start:end,:], atol=1e-2, rtol=3e-2
        )


def _run_allgather_worker(world_size, rank, hidden_dim, dtype):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    device = torch.device(f"cuda:{rank}")
    mapp = MoE_Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
    )
    sizes = [world_size * (i + 1) for i in range(world_size)]
    total_size = sum(sizes)
    shape_ref = (total_size, hidden_dim)
        
    out_ref = torch.randn(shape_ref, dtype=dtype, device=device)
    start = sum(sizes[:rank])
    end = start + sizes[rank]
    inp = out_ref[start:end, :]
    out = comm.all_gather(
        inp,
        mapp,
        dim=0,
        sizes=sizes,
    )
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=3e-2)


@pytest.mark.parametrize("hidden_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trtllm_all_gather(hidden_dim, dtype):
    torch.manual_seed(RANDOM_SEED)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.size
    assert world_size > 0
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    _run_allgather_worker(world_size, rank, hidden_dim, dtype)

    print(f"all_gather with tp = {world_size}: OK")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trtllm_reduce_scatter(dtype):
    torch.manual_seed(RANDOM_SEED)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.size
    assert world_size > 0
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    _run_reduce_scatter_worker(rank, world_size, dtype)

    print(f"reduce_scatter with tp = {world_size}: OK")
