"""Resolve EP comm group rank/world from :class:`BootstrapConfig`."""

from __future__ import annotations

import hashlib
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


# Per-GROUP generation counters namespacing derived rendezvous stores, keyed by
# (subsystem, sorted global-rank tuple). Fleet creation is collective over the
# EP group, so each group's counter agrees across its ranks; re-created fleets
# then never reuse a prior fleet's keys. A single process-wide counter would
# diverge when a process belongs to several EP subgroups and creates their
# fleets in a different interleaving than its peers.
_STORE_GENS: dict = {}


def resolve_rendezvous_store(bootstrap: BootstrapConfig, *, subsystem: str):
    """Return a rendezvous store for ``subsystem``, derived from ``bootstrap``.

    Resolution order (the transport-agnostic analogue of nccl_ep's
    ``_resolve_comm``):

    1. ``bootstrap.tcp_store`` set — use it as-is.
    2. otherwise — derive a ``PrefixStore`` from torch.distributed's default
       store, so hosts that pass only ``process_group`` (e.g. vLLM's EP group)
       work without constructing a second TCPStore on a sibling port. The
       prefix is namespaced by ``subsystem``, the EP group's global ranks, and
       that group's generation counter, so disjoint EP subgroups, different
       subsystems, and re-created fleets never collide on store keys.

    Used by the NIXL ``Buffer`` bootstrap (``subsystem="nixl_ep"``) and by
    fault-tolerance mask reconciliation (``subsystem="ft"``), which needs a
    store on both transports.

    Note for callers: each call bumps the generation counter, so resolve ONCE
    and cache the result. Resolving per operation would put each rank on a
    different prefix.
    """
    if bootstrap.tcp_store is not None:
        return bootstrap.tcp_store

    import torch.distributed as dist

    if not dist.is_initialized():
        raise ValueError(
            f"{subsystem} needs a rendezvous store: set bootstrap.tcp_store "
            "(a torch.distributed.TCPStore), or initialize torch.distributed "
            "so one can be derived from the default store."
        )

    # torch exposes no public accessor for the default store;
    # _get_default_store has been stable across torch 2.x but is private, so
    # fail with the explicit-tcp_store escape hatch rather than a raw
    # AttributeError if it ever moves.
    try:
        base_store = dist.distributed_c10d._get_default_store()
    except (AttributeError, RuntimeError) as e:
        raise ValueError(
            "Could not derive a rendezvous store from torch.distributed's "
            "default store; set bootstrap.tcp_store (a "
            "torch.distributed.TCPStore) explicitly."
        ) from e

    ranks = tuple(sorted(dist.get_process_group_ranks(bootstrap_comm_group(bootstrap))))
    key = (subsystem, ranks)
    gen = _STORE_GENS.get(key, 0)
    _STORE_GENS[key] = gen + 1
    # Encode the FULL group identity: a min×len-style prefix collides for
    # overlapping groups like (0,1,2,3) vs (0,2,4,6). Digest the rank tuple
    # (not hash(), which is per-process randomized) to keep the prefix bounded
    # for large groups while staying identical across ranks.
    group_id = hashlib.sha1("-".join(map(str, ranks)).encode()).hexdigest()[:12]
    prefix = f"flashinfer/moe_ep/{subsystem}/{group_id}/{gen}"
    return dist.PrefixStore(prefix, base_store)


__all__ = [
    "bootstrap_comm_group",
    "bootstrap_ep_rank_world",
    "bootstrap_ep_world_size",
    "resolve_rendezvous_store",
]
