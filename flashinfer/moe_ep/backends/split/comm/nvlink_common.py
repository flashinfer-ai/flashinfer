"""Shared setup for the NVLink symmetric-memory communication backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .....comm.abstractions import CommBackend
    from .....comm.mapping import Mapping
    from .....comm.mnnvl import MnnvlConfig, MnnvlMemory
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


def unmap_mnnvl_memory(
    memory: "MnnvlMemory", before_release: Optional[Callable[[], Any]] = None
) -> bool:
    """Release the physical backing of ``memory`` on every rank while keeping
    its virtual addresses reserved, e.g. so that the process can be
    checkpointed.

    Collective over the allocation's communicator once every rank has quiesced
    its CUDA work. ``before_release`` runs after that point and before the
    memory is unmapped, for resources bound to it. Returns ``False``, without
    communicating, when ``memory`` is already unmapped.
    """
    import torch

    from .....comm.mnnvl import MnnvlMemory

    record = MnnvlMemory.allocated_map[memory.ptr]
    if not record.mapped:
        return False
    # Every rank must stop using the memory before any rank releases it.
    torch.cuda.synchronize()
    record.comm.barrier()
    if before_release is not None:
        before_release()
    MnnvlMemory._unmap_and_release_handles(record)
    record.mem_handles = [None] * record.comm_size
    record.mapped = False
    # Do not return until every rank has released its backing.
    record.comm.barrier()
    return True


def remap_mnnvl_memory(memory: "MnnvlMemory", comm_backend: "CommBackend") -> bool:
    """Back the reserved virtual addresses of ``memory``, released by
    :func:`unmap_mnnvl_memory`, with new physical memory whose handles are
    exchanged over ``comm_backend``. The new memory is uninitialized.

    Collective over ``comm_backend``, which must have the rank and size of the
    original communicator; it replaces that communicator. Returns ``False``,
    without communicating, when ``memory`` is mapped.
    """
    import torch

    from .....comm.mnnvl import MnnvlMemory

    record = MnnvlMemory.allocated_map[memory.ptr]
    if record.mapped:
        return False
    comm_size = comm_backend.Get_size()
    comm_rank = comm_backend.Get_rank()
    if comm_size != record.comm_size or comm_rank != record.comm_rank:
        raise RuntimeError(
            "Cannot remap MNNVL memory over a communicator with rank/size "
            f"{comm_rank}/{comm_size}; it was allocated over "
            f"{record.comm_rank}/{record.comm_size}"
        )
    torch.cuda.synchronize()
    record.mem_handles = MnnvlMemory._create_and_map_handles(
        comm_backend,
        record.aligned_size,
        record.start_address,
        record.rank_stride,
        record.address_offset,
    )
    record.comm = comm_backend
    MnnvlMemory.comm = comm_backend
    record.mapped = True
    return True
