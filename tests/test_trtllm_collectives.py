# import multiprocessing as mp
# import socket
from typing import Any

import pytest
import torch
# import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.mapping import Mapping

RANDOM_SEED = 42

def _run_reduce_scatter_worker(rank, world_size, dtype):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    mapp = Mapping(
				world_size=world_size,
				rank=rank,
				gpus_per_node=world_size,
				tp_size=world_size,
		)

		# Create input tensor with shape (world_size, world_size)
		# Each rank will have a different slice of this tensor
    shape = (world_size, world_size)
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    rank_slice = input_tensor[rank, :]
    output = comm.reduce_scatter(
				rank_slice,
				mapp,
				dim=-1,
		)

    expected_output = input_tensor.sum(dim=0)
    torch.testing.assert_close(
				output[0], expected_output[rank], atol=1e-3, rtol=3e-2
		)


def _run_allgather_worker(world_size, rank, hidden_dim, dtype):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    device = torch.device(f"cuda:{rank}")
    mapp = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
    )
    shape_ref = (world_size, world_size, hidden_dim)
    out_ref = torch.randn(shape_ref, dtype=dtype, device=device)
    inp = out_ref[rank,:,:]
    out = comm.all_gather(
        inp,
        mapp,
        dim=0,
    ).reshape(shape_ref)
    torch.testing.assert_close(
        out, out_ref, atol=1e-3, rtol=3e-2
    )


@pytest.mark.mpi
@pytest.mark.parametrize("hidden_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trtllm_all_gather(hidden_dim, dtype):
    torch.manual_seed(RANDOM_SEED)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.size
    assert world_size> 0
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    _run_allgather_worker(world_size, rank, hidden_dim, dtype)

    print(f"all_gather with tp = {world_size}: OK")


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trtllm_reduce_scatter(dtype):
    torch.manual_seed(RANDOM_SEED)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.size
    assert world_size> 0
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    _run_reduce_scatter_worker(rank, world_size, dtype)

    print(f"reduce_scatter with tp = {world_size}: OK")