"""Shared setup for the NVLink symmetric-memory communication backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .....comm.comm_backend import CommBackend
    from .....comm.mapping import Mapping
    from .....comm.mnnvl import MnnvlConfig
    from ....config import BootstrapConfig

logger = logging.getLogger(__name__)


def nvlink_platform_supported() -> bool:
    """Whether this rank can run the NVLink backends; logs the reason if not.

    The rank needs CUDA and must meet :meth:`MnnvlMemory.supports_mnnvl`, i.e.
    its GPU has P2P-capable NVLinks and all of them are up.
    """
    try:
        import pynvml
        import torch
    except ImportError as exc:
        logger.error("NVLink backends unavailable: %s", exc)
        return False
    if not torch.cuda.is_available():
        logger.error("NVLink backends unavailable: CUDA is not available")
        return False
    try:
        from .....comm.mnnvl import MnnvlMemory

        supported = bool(MnnvlMemory.supports_mnnvl())
    except pynvml.NVMLError as exc:
        logger.error("NVLink backends unavailable: NVML query failed: %s", exc)
        return False
    except (ImportError, RuntimeError, OSError) as exc:
        logger.error("NVLink backends unavailable: %s: %s", type(exc).__name__, exc)
        return False
    if not supported:
        logger.error(
            "NVLink backends unavailable: GPU %d has no P2P-capable NVLink or "
            "not all of its NVLinks are up",
            torch.cuda.current_device(),
        )
    return supported


def mnnvl_mapping_and_config(
    bootstrap: "BootstrapConfig",
    comm_backend: Optional["CommBackend"] = None,
) -> tuple["Mapping", "MnnvlConfig"]:
    """Describe the EP group as a pure-EP :class:`Mapping` plus the communicator
    used to exchange MNNVL memory handles.

    ``comm_backend`` defaults to torch.distributed over
    ``bootstrap.process_group`` (the default group when unset), which must
    already be initialized.

    Collective: every rank reports whether it can run the NVLink backends,
    and all ranks raise if any cannot, so an unsupported rank never fails alone
    while its peers wait in the symmetric-memory setup. Each unsupported rank
    logs its own reason.
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
    verdicts = comm_backend.allgather(nvlink_platform_supported())
    unsupported = [rank for rank, supported in enumerate(verdicts) if not supported]
    if unsupported:
        raise RuntimeError(
            f"NVLink backends are not supported on EP ranks {unsupported}; see "
            "their logs for the reason"
        )
    mapping = Mapping(
        world_size=bootstrap.world_size,
        rank=bootstrap.rank,
        gpus_per_node=bootstrap.world_size,
        tp_size=bootstrap.world_size,
        moe_ep_size=bootstrap.world_size,
    )
    return mapping, MnnvlConfig(comm_backend=comm_backend)
