"""Singleton metadata for an existing caller-owned RoCE communicator.

Declaration and host enumeration do not discover HCAs, connect a proxy, or
qualify remote topology. In particular, two local GPUs provide no evidence of
remote RoCE support. Peer hosts and HCA selection remain caller metadata.
"""

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, FrozenMapping, make_fixed_contract


SURFACES = ("AllReduce.all_reduce", "AllReduce.all_gather")


@dataclass(frozen=True, kw_only=True)
class RoceQuery:
    surface: str
    world_size: int
    rank: int
    topology: str
    peer_hosts: tuple[str, ...]
    hca_names: tuple[str, ...]
    # Tensor geometry, dtype, gather layout and communicator capacity/launch
    # settings are supplied verbatim; no transport or precision race is added.
    call: FrozenMapping
    setup: FrozenMapping


RoceConfig = BackendConfig
TUNING = make_fixed_contract(
    component_id="comm.roce",
    query_type=RoceQuery,
    backend="native",
)
_validate_fixed_query = TUNING.validate_query


def _validate_query(query: RoceQuery, device) -> None:
    _validate_fixed_query(query, device)
    if query.surface not in SURFACES:
        raise ValueError(f"unknown RoCE execution surface {query.surface!r}")
    if query.world_size not in range(2, 17):
        raise ValueError("RoCE supports world sizes 2 through 16")
    if not 0 <= query.rank < query.world_size:
        raise ValueError("rank must belong to the caller's process group")
    if query.topology != "roce_rdma":
        raise ValueError("RoCE requires caller-established RDMA topology")
    if len(query.peer_hosts) != query.world_size or not all(query.peer_hosts):
        raise ValueError("peer_hosts must identify every rank's host")
    # An empty HCA selection is legal: the production constructor discovers
    # active devices. Neither explicit names nor discovery imply qualification.


TUNING = replace(TUNING, validate_query=_validate_query)

__all__ = ["RoceQuery", "RoceConfig", "SURFACES", "TUNING"]
