# Helper functions for communication tests
import os

import pytest
import torch
import torch.distributed as dist


def _get_rank_info_from_env():
    """Get rank and world_size from environment variables.

    Supports multiple launchers:
    - SLURM (srun): SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID
    - torchrun: RANK, WORLD_SIZE, LOCAL_RANK
    - MPI (fallback): Uses mpi4py if environment variables not found

    Returns:
        tuple: (rank, world_size, local_rank)
    """
    # Try SLURM environment variables first (set by srun)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(
            os.environ.get("SLURM_LOCALID", rank % torch.cuda.device_count())
        )
        return rank, world_size, local_rank

    # Try torchrun/torch.distributed.launch environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        return rank, world_size, local_rank

    # Fallback to MPI if available
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        local_rank = rank % torch.cuda.device_count()
        return rank, world_size, local_rank
    except ImportError as e:
        raise RuntimeError(
            "Could not determine rank/world_size. "
            "Please set SLURM_PROCID/SLURM_NTASKS (srun), "
            "RANK/WORLD_SIZE (torchrun), or install mpi4py."
        ) from e


def setup_mpi_and_cuda():
    """Setup distributed environment and CUDA device for tests.

    Returns:
        tuple: (rank, world_size, gpus_per_node)

    Raises:
        pytest.skip: If no CUDA devices or fewer than 2 ranks
    """
    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node == 0:
        pytest.skip("Tests require at least one CUDA device per node")

    rank, world_size, local_rank = _get_rank_info_from_env()

    if world_size < 2:
        pytest.skip(f"Tests require at least 2 ranks, got {world_size}")

    torch.cuda.set_device(local_rank)

    return rank, world_size, gpus_per_node


def _get_master_addr():
    """Get the master address for torch.distributed.

    For multi-node SLURM jobs, extracts the first node from SLURM_NODELIST.
    """
    if "MASTER_ADDR" in os.environ:
        return os.environ["MASTER_ADDR"]

    # For SLURM multi-node: get first node from nodelist
    if "SLURM_NODELIST" in os.environ:
        import subprocess

        try:
            # Use scontrol to expand the nodelist and get first node
            result = subprocess.run(
                ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
                capture_output=True,
                text=True,
                check=True,
            )
            first_node = result.stdout.strip().split("\n")[0]
            return first_node
        except (subprocess.CalledProcessError, FileNotFoundError):
            # scontrol not available, try simple parsing
            nodelist = os.environ["SLURM_NODELIST"]
            # Handle simple cases like "node[0-3]" -> "node0" or "node0,node1" -> "node0"
            if "[" in nodelist:
                base = nodelist.split("[")[0]
                nums = nodelist.split("[")[1].split("]")[0]
                first_num = nums.split(",")[0].split("-")[0]
                return f"{base}{first_num}"
            else:
                return nodelist.split(",")[0]

    return "localhost"


def init_torch_distributed_from_mpi():
    """Initialize torch.distributed using environment rank info.

    Supports SLURM (srun), torchrun, or MPI launchers.
    Safe to call multiple times - will skip if already initialized.

    Uses explicit init_method to avoid modifying environment variables.
    """
    if dist.is_initialized():
        return

    rank, world_size, local_rank = _get_rank_info_from_env()

    # Use explicit init_method instead of setting environment variables
    master_addr = _get_master_addr()
    master_port = os.environ.get("MASTER_PORT", "29500")
    init_method = f"tcp://{master_addr}:{master_port}"

    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def cleanup_torch_distributed():
    """Cleanup torch.distributed if initialized.

    Safe to call even if torch.distributed was not initialized.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
