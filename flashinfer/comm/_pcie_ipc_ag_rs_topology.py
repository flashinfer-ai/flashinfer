"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import hashlib
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


@dataclass
class _RankLinks:
    rank: int
    hostname: str = ""
    device_uuid: str = ""
    peer_system: Dict[str, bool] = field(default_factory=dict)
    pair_errors: Dict[str, str] = field(default_factory=dict)
    probe_error: Optional[str] = None


@dataclass(frozen=True)
class PcieIpcAgRsTopology:
    ordered_4plus4: bool
    reason: str
    placement_fingerprint: str


def _identity(rank: int, device: torch.device) -> _RankLinks:
    result = _RankLinks(rank=rank)
    try:
        result.hostname = socket.gethostname()
        properties = torch.cuda.get_device_properties(device)
        uuid = getattr(properties, "uuid", None)
        if uuid is None:
            raise RuntimeError("CUDA device UUID is unavailable")
        result.device_uuid = f"GPU-{uuid}"
    except Exception as err:  # noqa: BLE001 - unknown topology fails closed
        result.probe_error = f"{type(err).__name__}: {err}"
    return result


def _probe_links(identity: _RankLinks, peer_uuids: List[str]) -> _RankLinks:
    result = _RankLinks(
        rank=identity.rank,
        hostname=identity.hostname,
        device_uuid=identity.device_uuid,
        probe_error=identity.probe_error,
    )
    if result.probe_error:
        return result
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            source = pynvml.nvmlDeviceGetHandleByUUID(result.device_uuid.encode())
            for peer_uuid in peer_uuids:
                if peer_uuid == result.device_uuid:
                    continue
                try:
                    peer = pynvml.nvmlDeviceGetHandleByUUID(peer_uuid.encode())
                    ancestor = pynvml.nvmlDeviceGetTopologyCommonAncestor(source, peer)
                    result.peer_system[peer_uuid] = (
                        ancestor >= pynvml.NVML_TOPOLOGY_SYSTEM
                    )
                except pynvml.NVMLError as err:
                    result.pair_errors[peer_uuid] = str(err)
        finally:
            pynvml.nvmlShutdown()
    except Exception as err:  # noqa: BLE001 - unknown topology fails closed
        result.probe_error = f"{type(err).__name__}: {err}"
    return result


def _fingerprint(topologies: List[_RankLinks]) -> str:
    records = [
        (
            topology.rank,
            topology.hostname,
            topology.device_uuid,
            tuple(sorted(topology.peer_system.items())),
            tuple(sorted(topology.pair_errors.items())),
            topology.probe_error,
        )
        for topology in sorted(topologies, key=lambda item: item.rank)
    ]
    return hashlib.sha256(repr(records).encode()).hexdigest()[:16]


def decide_pcie_ipc_ag_rs_topology(
    topologies: List[_RankLinks],
) -> PcieIpcAgRsTopology:
    """Admit the TP8 4+4 schedule only for its measured logical rank order."""
    fingerprint = _fingerprint(topologies)
    if len(topologies) != 8:
        return PcieIpcAgRsTopology(
            False, f"requires 8 ranks, got {len(topologies)}", fingerprint
        )

    by_rank = {topology.rank: topology for topology in topologies}
    if set(by_rank) != set(range(8)):
        return PcieIpcAgRsTopology(
            False, "rank evidence must contain logical ranks 0 through 7", fingerprint
        )
    failed = [topology.rank for topology in topologies if topology.probe_error]
    if failed:
        return PcieIpcAgRsTopology(
            False, f"topology probe failed on ranks {failed}", fingerprint
        )

    hosts = {topology.hostname for topology in topologies}
    uuids = [by_rank[rank].device_uuid for rank in range(8)]
    if len(hosts) != 1:
        return PcieIpcAgRsTopology(False, "ranks do not share one host", fingerprint)
    if any(not uuid for uuid in uuids) or len(set(uuids)) != 8:
        return PcieIpcAgRsTopology(
            False, "rank GPU UUIDs must be present and unique", fingerprint
        )

    for left in range(8):
        for right in range(left + 1, 8):
            left_uuid = uuids[left]
            right_uuid = uuids[right]
            left_links = by_rank[left]
            right_links = by_rank[right]
            if (
                right_uuid in left_links.pair_errors
                or left_uuid in right_links.pair_errors
            ):
                return PcieIpcAgRsTopology(
                    False, f"rank pair ({left}, {right}) was not probeable", fingerprint
                )
            forward = left_links.peer_system.get(right_uuid)
            reverse = right_links.peer_system.get(left_uuid)
            if forward is None or reverse is None or forward != reverse:
                return PcieIpcAgRsTopology(
                    False,
                    f"rank pair ({left}, {right}) has incomplete or asymmetric evidence",
                    fingerprint,
                )
            expected = (left < 4) != (right < 4)
            if forward != expected:
                return PcieIpcAgRsTopology(
                    False,
                    f"rank pair ({left}, {right}) does not match logical 4+4 islands",
                    fingerprint,
                )

    return PcieIpcAgRsTopology(
        True,
        "logical ranks 0..3 and 4..7 form two system-local islands",
        fingerprint,
    )


def resolve_pcie_ipc_ag_rs_topology(
    group: ProcessGroup, device: torch.device
) -> PcieIpcAgRsTopology:
    """Collect exact rank UUIDs, then query TP8 peer links through NVML."""
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    local_identity = _identity(rank, device)
    identities: List[Optional[_RankLinks]] = [None] * world_size
    dist.all_gather_object(identities, local_identity, group=group)
    gathered_identities = [identity for identity in identities if identity is not None]
    if world_size != 8:
        return decide_pcie_ipc_ag_rs_topology(gathered_identities)

    peer_uuids = [
        identity.device_uuid for identity in gathered_identities if identity.device_uuid
    ]
    local_links = _probe_links(local_identity, peer_uuids)
    links: List[Optional[_RankLinks]] = [None] * world_size
    dist.all_gather_object(links, local_links, group=group)
    return decide_pcie_ipc_ag_rs_topology(
        [topology for topology in links if topology is not None]
    )


__all__ = [
    "PcieIpcAgRsTopology",
    "decide_pcie_ipc_ag_rs_topology",
    "resolve_pcie_ipc_ag_rs_topology",
]
