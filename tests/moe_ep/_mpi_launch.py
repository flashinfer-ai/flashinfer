"""Launch helper shared by the multi-rank moe_ep tests."""

from __future__ import annotations

import os

# Set by Open MPI, MPICH/Hydra and PMIx launchers respectively.
_MPI_LAUNCH_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK")


def adopt_mpi_launch() -> None:
    """Set torch.distributed's ``env://`` variables from MPI when an MPI
    launcher started this process instead of torchrun."""
    if "WORLD_SIZE" in os.environ or not any(
        var in os.environ for var in _MPI_LAUNCH_VARS
    ):
        return
    import socket

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    master = None
    if comm.Get_rank() == 0:
        with socket.socket() as sock:
            sock.bind(("", 0))
            master = (socket.gethostname(), sock.getsockname()[1])
    master_addr, master_port = comm.bcast(master, root=0)
    local_rank = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    os.environ.update(
        RANK=str(comm.Get_rank()),
        WORLD_SIZE=str(comm.Get_size()),
        LOCAL_RANK=str(local_rank),
        MASTER_ADDR=master_addr,
        MASTER_PORT=str(master_port),
    )
