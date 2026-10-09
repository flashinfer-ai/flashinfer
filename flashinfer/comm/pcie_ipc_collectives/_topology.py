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

Physical GPU identities and NVML evidence shared by PCIe IPC collectives.
"""

import socket
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


@dataclass
class PcieIpcTopologyEvidence:
    rank: int
    hostname: str = ""
    device_index: int = -1
    device_uuid: str = ""
    peer_ancestors: Dict[str, int] = field(default_factory=dict)
    pair_errors: Dict[str, str] = field(default_factory=dict)
    probe_error: Optional[str] = None
    # Carry NVML's own boundaries so pure decisions need neither NVML nor
    # hardcoded copies of its enum values.
    hostbridge_level: Optional[int] = None
    system_level: Optional[int] = None


def _device_uuid(device: int) -> str:
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    if uuid is None:
        raise RuntimeError("CUDA device UUID is unavailable")
    return f"GPU-{uuid}"


def probe_pcie_ipc_identity(
    rank: int, device: Optional[torch.device] = None
) -> PcieIpcTopologyEvidence:
    result = PcieIpcTopologyEvidence(rank=rank)
    try:
        result.hostname = socket.gethostname()
        parsed = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        if parsed.type != "cuda":
            raise ValueError(f"probe requires a CUDA device, got {parsed!r}")
        result.device_index = (
            parsed.index if parsed.index is not None else torch.cuda.current_device()
        )
        result.device_uuid = _device_uuid(result.device_index)
    except Exception as err:  # noqa: BLE001 - exchange failures as evidence
        result.probe_error = f"{type(err).__name__}: {err}"
    return result


def probe_pcie_ipc_links(
    identity: PcieIpcTopologyEvidence, peer_uuids: Optional[List[str]] = None
) -> PcieIpcTopologyEvidence:
    """Query exact peers, or visible GPUs for the standalone AR probe API."""
    result = replace(identity, peer_ancestors={}, pair_errors={})
    if result.probe_error:
        return result
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            result.hostbridge_level = pynvml.NVML_TOPOLOGY_HOSTBRIDGE
            result.system_level = pynvml.NVML_TOPOLOGY_SYSTEM
            source = pynvml.nvmlDeviceGetHandleByUUID(result.device_uuid.encode())
            if peer_uuids is None:
                peer_uuids = [
                    _device_uuid(index) for index in range(torch.cuda.device_count())
                ]
            for peer_uuid in peer_uuids:
                if peer_uuid == result.device_uuid:
                    continue
                try:
                    peer = pynvml.nvmlDeviceGetHandleByUUID(peer_uuid.encode())
                    result.peer_ancestors[peer_uuid] = (
                        pynvml.nvmlDeviceGetTopologyCommonAncestor(source, peer)
                    )
                except pynvml.NVMLError as err:
                    result.pair_errors[peer_uuid] = str(err)
        finally:
            pynvml.nvmlShutdown()
    except Exception as err:  # noqa: BLE001 - unknown evidence stays unknown
        result.probe_error = f"{type(err).__name__}: {err}"
    return result


def collect_pcie_ipc_topology(
    group: ProcessGroup,
    device: Optional[torch.device] = None,
    *,
    probe_links: bool = True,
) -> List[PcieIpcTopologyEvidence]:
    """Exchange identities before probing, independent of CUDA visibility order."""
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    local_identity = probe_pcie_ipc_identity(rank, device)
    identities: List[Optional[PcieIpcTopologyEvidence]] = [None] * world_size
    dist.all_gather_object(identities, local_identity, group=group)
    gathered_identities = [identity for identity in identities if identity is not None]
    if not probe_links:
        return gathered_identities

    peer_uuids = [
        identity.device_uuid for identity in gathered_identities if identity.device_uuid
    ]
    local_links = probe_pcie_ipc_links(local_identity, peer_uuids)
    links: List[Optional[PcieIpcTopologyEvidence]] = [None] * world_size
    dist.all_gather_object(links, local_links, group=group)
    return [topology for topology in links if topology is not None]
