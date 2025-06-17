# import multiprocessing as mp
# import socket
from typing import Any

import pytest
import torch

import flashinfer.comm as comm
from flashinfer import MoE_Mapping

# import torch.distributed as dist


RANDOM_SEED = 42
NUM_REPEATS = 8


def _run_reduce_scatter_worker(rank, world_size, dtype):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    mapp = MoE_Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
    )

    # Create input tensor with shape (world_size, world_size)
    # Each rank will have a different slice of this tensor
    shape = (world_size, world_size)
    input_tensors = [
        torch.randn(shape, dtype=dtype, device=device) for _ in range(NUM_REPEATS)
    ]
    expected_output = [i.sum(dim=0) for i in input_tensors]

    output = comm.reduce_scatter(
        [i[rank, :] for i in input_tensors],
        mapp,
        dim=-1,
    )

    for i in range(NUM_REPEATS):
        torch.testing.assert_close(
            output[i][0], expected_output[i][rank], atol=1e-3, rtol=3e-2
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
    shape_ref = (world_size, world_size, hidden_dim)
    out_ref = torch.randn(shape_ref, dtype=dtype, device=device)
    inp = out_ref[rank, :, :]
    out = comm.all_gather(
        inp,
        mapp,
        dim=0,
    ).reshape(shape_ref)
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
