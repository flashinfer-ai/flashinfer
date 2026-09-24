"""Shared setup for the NVLink symmetric-memory communication backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .....comm.comm_backend import CommBackend
    from .....comm.mapping import Mapping
    from .....comm.mnnvl import MnnvlConfig
    from ....config import BootstrapConfig


def nvlink_platform_supported() -> bool:
    """Whether this GPU meets :meth:`MnnvlMemory.supports_mnnvl` (all NVLinks up)."""
    try:
        import pynvml
        import torch

        if not torch.cuda.is_available():
            return False
        from .....comm.mnnvl import MnnvlMemory

        return bool(MnnvlMemory.supports_mnnvl())
    except ImportError:
        return False
    except (RuntimeError, OSError, pynvml.NVMLError):
        return False


def mnnvl_mapping_and_config(
    bootstrap: "BootstrapConfig",
    comm_backend: Optional["CommBackend"] = None,
) -> tuple["Mapping", "MnnvlConfig"]:
    """Describe the EP group as a pure-EP :class:`Mapping` plus the communicator
    used to exchange MNNVL memory handles.

    ``comm_backend`` defaults to torch.distributed over
    ``bootstrap.process_group`` (the default group when unset), which must
    already be initialized.
    """
    from .....comm.comm_backend import TorchDistBackend
    from .....comm.mapping import Mapping
    from .....comm.mnnvl import MnnvlConfig

    if comm_backend is None:
        comm_backend = TorchDistBackend(group=bootstrap.process_group)
    if comm_backend.Get_size() != bootstrap.world_size:
        raise ValueError(
            f"MNNVL communicator has {comm_backend.Get_size()} ranks but the EP "
            f"group has world_size={bootstrap.world_size}"
        )
    if comm_backend.Get_rank() != bootstrap.rank:
        raise ValueError(
            f"MNNVL communicator rank {comm_backend.Get_rank()} does not match "
            f"the EP rank {bootstrap.rank}"
        )
    mapping = Mapping(
        world_size=bootstrap.world_size,
        rank=bootstrap.rank,
        gpus_per_node=bootstrap.world_size,
        tp_size=bootstrap.world_size,
        moe_ep_size=bootstrap.world_size,
    )
    return mapping, MnnvlConfig(comm_backend=comm_backend)
