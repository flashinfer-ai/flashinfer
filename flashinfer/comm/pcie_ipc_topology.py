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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.distributed import ProcessGroup

from .pcie_ipc_collectives._topology import (
    PcieIpcTopologyEvidence,
    collect_pcie_ipc_topology,
    probe_pcie_ipc_identity,
    probe_pcie_ipc_links,
)

# Which fabric the group is on. The distinction is the interconnect, not the
# GPU: the same card behaves differently depending on whether its NUMA island
# contains a PCIe switch. The profile keys the tune cache and screens CE island
# candidates; it does not establish an ordered 4+4 rank placement.
PROFILE_ROOTCPLX = "rootcplx-noswitch"
PROFILE_SWITCHPAIR = "pcieswitch-pairs"
PCIE_IPC_PROFILES = (PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR)

_PROFILE_ALIASES = {
    "rootcplx": PROFILE_ROOTCPLX,
    "rootcplx-noswitch": PROFILE_ROOTCPLX,
    "pcieswitch": PROFILE_SWITCHPAIR,
    "pcieswitch-pairs": PROFILE_SWITCHPAIR,
}


@dataclass
class PcieIpcRankTopology:
    """Per-rank probe result, exchanged across the group.

    ``peer_switch_local`` is keyed by the *peer GPU's UUID* so the decision
    layer can join results across ranks regardless of each process's
    ``CUDA_VISIBLE_DEVICES`` ordering. The collective resolver queries group
    members even when their GPUs are hidden from this process's CUDA view.
    """

    rank: int
    hostname: str = ""
    device_index: int = -1
    device_uuid: str = ""
    peer_switch_local: Dict[str, bool] = field(default_factory=dict)
    pair_errors: Dict[str, str] = field(default_factory=dict)
    probe_error: Optional[str] = None


@dataclass(frozen=True)
class PcieIpcProfileDecision:
    profile: str
    reason: str


def _profile_links(topology: PcieIpcTopologyEvidence) -> PcieIpcRankTopology:
    hostbridge = topology.hostbridge_level
    return PcieIpcRankTopology(
        rank=topology.rank,
        hostname=topology.hostname,
        device_index=topology.device_index,
        device_uuid=topology.device_uuid,
        peer_switch_local=(
            {
                uuid: level < hostbridge
                for uuid, level in topology.peer_ancestors.items()
            }
            if hostbridge is not None
            else {}
        ),
        pair_errors=topology.pair_errors,
        probe_error=topology.probe_error,
    )


def probe_pcie_ipc_rank_topology(
    rank: int, device: Optional[torch.device] = None
) -> PcieIpcRankTopology:
    """Probe this rank's GPU against other visible GPUs, recording failures.

    The collective resolver exchanges group UUIDs first so it can also query
    peers hidden by this process's ``CUDA_VISIBLE_DEVICES``.
    """
    return _profile_links(probe_pcie_ipc_links(probe_pcie_ipc_identity(rank, device)))


def decide_pcie_ipc_profile(
    requested: Optional[str], topologies: List[PcieIpcRankTopology]
) -> PcieIpcProfileDecision:
    """Pick the fabric label from the gathered probes. Pure function.

    An explicit ``requested`` profile always wins. Otherwise the group is
    switch-paired only if some rank positively observed a switch-local peer;
    anything unknown or unprobeable falls back to ``rootcplx-noswitch``.
    The profile keys the tuning cache and screens CE island candidates, so
    the label that claims less is the safe default.
    """
    # The intra-node constraint is checked first: CUDA IPC cannot cross hosts,
    # so an explicit profile must not be able to wave it through.
    hosts = {t.hostname for t in topologies if t.hostname}
    if len(hosts) > 1:
        raise ValueError(
            f"pcie ipc all-reduce is intra-node only, but the group spans {sorted(hosts)}"
        )

    if requested is not None:
        key = requested.strip().lower()
        if key not in _PROFILE_ALIASES:
            raise ValueError(
                f"unknown pcie ipc profile {requested!r}; "
                f"expected one of {sorted(_PROFILE_ALIASES)}"
            )
        return PcieIpcProfileDecision(_PROFILE_ALIASES[key], "requested explicitly")

    failed = [t.rank for t in topologies if t.probe_error]
    if failed:
        return PcieIpcProfileDecision(
            PROFILE_ROOTCPLX, f"probe failed on ranks {failed}; assuming no switch pair"
        )

    # Only pairs where BOTH endpoints belong to this group count. Standalone
    # probes can include visible GPUs outside the group; the collective
    # resolver already queries exact group members. An outside switch-local
    # pair says nothing about how the group's own ranks talk to each other.
    members = {t.device_uuid for t in topologies if t.device_uuid}
    for t in topologies:
        for peer_uuid, switch_local in t.peer_switch_local.items():
            if switch_local and peer_uuid in members:
                return PcieIpcProfileDecision(
                    PROFILE_SWITCHPAIR,
                    f"rank {t.rank} shares a PCIe switch with group member {peer_uuid}",
                )

    partial = [t.rank for t in topologies if any(u in members for u in t.pair_errors)]
    if partial:
        return PcieIpcProfileDecision(
            PROFILE_ROOTCPLX,
            f"some in-group pairs unprobeable on ranks {partial}; "
            "assuming no switch pair",
        )
    return PcieIpcProfileDecision(PROFILE_ROOTCPLX, "no switch-local pair observed")


def resolve_pcie_ipc_profile(
    group: ProcessGroup,
    requested: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> PcieIpcProfileDecision:
    """Probe every rank and agree on one profile.

    Collective. Runs before any workspace allocation or JIT build so an
    unsupported topology costs nothing, and gathers the per-rank probes so
    every rank reaches the same decision from the same evidence.
    """
    gathered = collect_pcie_ipc_topology(group, device)
    return decide_pcie_ipc_profile(requested, [_profile_links(t) for t in gathered])
