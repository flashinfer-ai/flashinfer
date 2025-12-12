# Helper functions for communication tests
import os

import pytest
import torch
import torch.distributed as dist
from mpi4py import MPI


def setup_mpi_and_cuda():
    """Setup MPI and CUDA device for tests.

    Returns:
        tuple: (rank, world_size, gpus_per_node)

    Raises:
        pytest.skip: If no CUDA devices or fewer than 2 MPI ranks
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node == 0:
        pytest.skip("Tests require at least one CUDA device per node")
    if world_size < 2:
        pytest.skip(f"Tests require at least 2 MPI ranks, got {world_size}")

    local_rank = rank % gpus_per_node
    torch.cuda.set_device(local_rank)

    return rank, world_size, gpus_per_node


def init_torch_distributed_from_mpi():
    """Initialize torch.distributed using MPI rank info.

    This allows running torch.distributed operations within an MPI context.
    Safe to call multiple times - will skip if already initialized.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if dist.is_initialized():
        return

    # Set environment variables for torch.distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def cleanup_torch_distributed():
    """Cleanup torch.distributed if initialized.

    Safe to call even if torch.distributed was not initialized.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
