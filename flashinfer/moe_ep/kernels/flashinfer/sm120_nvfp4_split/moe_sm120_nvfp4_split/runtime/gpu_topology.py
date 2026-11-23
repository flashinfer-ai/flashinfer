"""Runtime GPU topology discovery for the package-local EP world."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple


@dataclass(frozen=True)
class GpuTopologyEntry:
    global_rank: int
    local_rank: int
    pci_bus_id: str
    numa_node: int
    # Host identity: PCI bus IDs and NUMA node numbers repeat across hosts, so
    # a GPU is (host, pci_bus_id) and a P2P domain is (host, numa_node).
    host: str = ""

    @property
    def p2p_domain(self) -> Tuple[str, int]:
        return (self.host, self.numa_node)


@dataclass(frozen=True)
class EpTransportTopology:
    peers: Tuple[GpuTopologyEntry, ...]
    same_numa_peer_count: int
    cross_numa_peer_count: int
    force_ibgda_peer_mask: int


def _check_cuda(result, operation: str, cuda_driver):
    error, *values = result
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with {error}")
    return values[0] if len(values) == 1 else tuple(values)


def discover_gpu_topology(local_rank: int, global_rank: int) -> GpuTopologyEntry:
    """Resolve the selected CUDA device to its physical PCI/NUMA identity."""

    from cuda.bindings import driver

    _check_cuda(driver.cuInit(0), "cuInit", driver)
    device = _check_cuda(
        driver.cuDeviceGet(local_rank), "cuDeviceGet", driver
    )
    raw_bdf = _check_cuda(
        driver.cuDeviceGetPCIBusId(32, device),
        "cuDeviceGetPCIBusId",
        driver,
    )
    pci_bus_id = raw_bdf.decode("ascii").rstrip("\x00 ").lower()
    numa_path = Path("/sys/bus/pci/devices") / pci_bus_id / "numa_node"
    try:
        numa_node = int(numa_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot resolve NUMA node for CUDA device {local_rank} "
            f"({pci_bus_id}) via {numa_path}"
        ) from exc
    if numa_node < 0:
        raise RuntimeError(
            f"CUDA device {local_rank} ({pci_bus_id}) has unknown NUMA node "
            f"{numa_node}; hybrid transport fails closed"
        )
    import socket

    return GpuTopologyEntry(
        global_rank=global_rank,
        local_rank=local_rank,
        pci_bus_id=pci_bus_id,
        numa_node=numa_node,
        host=socket.gethostname(),
    )


def gather_ep_transport_topology(
    local: GpuTopologyEntry,
    *,
    ep_rank: int,
    ep_control_group: object,
) -> EpTransportTopology:
    """Gather the actual EP device order and derive this PE's IBGDA mask."""

    import torch.distributed as dist

    ep_size = dist.get_world_size(group=ep_control_group)
    gathered = [None] * ep_size
    dist.all_gather_object(gathered, local, group=ep_control_group)
    return derive_ep_transport_topology(gathered, ep_rank=ep_rank)


def derive_ep_transport_topology(
    gathered: Sequence[GpuTopologyEntry], *, ep_rank: int
) -> EpTransportTopology:
    """Pure peer-order-to-mask transform used by bootstrap and unit tests."""

    peers = tuple(gathered)
    ep_size = len(peers)
    if any(not isinstance(peer, GpuTopologyEntry) for peer in peers):
        raise RuntimeError(f"invalid EP topology records: {peers!r}")
    if len({(peer.host, peer.pci_bus_id) for peer in peers}) != ep_size:
        raise RuntimeError(f"EP ranks do not map to unique GPUs: {peers!r}")
    if not 0 <= ep_rank < ep_size:
        raise RuntimeError(f"EP rank {ep_rank} is outside [0, {ep_size})")

    # Only GPUs on the same host and NUMA node are P2P (LSA) peers; every
    # other peer, cross-NUMA or on another host, goes through the GIN network.
    own_domain = peers[ep_rank].p2p_domain
    force_mask = 0
    same_numa = 0
    cross_numa = 0
    for peer_rank, peer in enumerate(peers):
        if peer_rank == ep_rank:
            continue
        if peer.p2p_domain == own_domain:
            same_numa += 1
        else:
            cross_numa += 1
            force_mask |= 1 << peer_rank
    return EpTransportTopology(
        peers=peers,
        same_numa_peer_count=same_numa,
        cross_numa_peer_count=cross_numa,
        force_ibgda_peer_mask=force_mask,
    )


def format_ep_topology(topology: EpTransportTopology) -> str:
    devices = ", ".join(
        f"pe{rank}={(peer.host + ':') if peer.host else ''}{peer.pci_bus_id}/numa{peer.numa_node}"
        for rank, peer in enumerate(topology.peers)
    )
    return (
        f"{devices}; same={topology.same_numa_peer_count}, "
        f"cross={topology.cross_numa_peer_count}, "
        f"force_ibgda_mask=0x{topology.force_ibgda_peer_mask:x}"
    )


__all__ = [
    "EpTransportTopology",
    "GpuTopologyEntry",
    "discover_gpu_topology",
    "derive_ep_transport_topology",
    "format_ep_topology",
    "gather_ep_transport_topology",
]
