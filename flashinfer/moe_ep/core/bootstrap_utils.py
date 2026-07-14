"""Resolve EP comm group rank/world from :class:`BootstrapConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import BootstrapConfig

if TYPE_CHECKING:
    import torch.distributed as dist


def bootstrap_comm_group(bootstrap: BootstrapConfig) -> "dist.ProcessGroup":
    """Return the process group EP mega/split comm should use.

    When ``BootstrapConfig.process_group`` is set (e.g. vLLM's EP group), that
    group is returned. Otherwise falls back to ``torch.distributed`` WORLD.
    """
    import torch.distributed as dist

    if bootstrap.process_group is not None:
        return bootstrap.process_group
    if not dist.is_initialized():
        raise RuntimeError(
            "MoEEpLayer requires torch.distributed to be initialized "
            "when BootstrapConfig.process_group is not set"
        )
    return dist.group.WORLD


def bootstrap_ep_rank_world(bootstrap: BootstrapConfig) -> tuple[int, int]:
    """Return ``(rank, world_size)`` within the EP comm group."""
    import torch.distributed as dist

    if bootstrap.process_group is not None and dist.is_initialized():
        pg = bootstrap.process_group
        return dist.get_rank(pg), dist.get_world_size(pg)
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return bootstrap.rank, bootstrap.world_size


def bootstrap_ep_world_size(bootstrap: BootstrapConfig) -> int:
    """Return EP comm group world size (``BootstrapConfig.world_size`` when validated)."""
    return bootstrap_ep_rank_world(bootstrap)[1]


__all__ = [
    "bootstrap_comm_group",
    "bootstrap_ep_rank_world",
    "bootstrap_ep_world_size",
]
