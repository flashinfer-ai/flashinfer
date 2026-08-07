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

import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# Which tuning table to use. The distinction is the interconnect, not the GPU:
# the same card behaves differently depending on whether its NUMA island
# contains a PCIe switch, and the tuned block counts and crossovers differ by
# substantially between the two.
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
    ``CUDA_VISIBLE_DEVICES`` ordering, and so the probe only ever describes
    GPUs this rank can actually see.
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


def probe_pcie_ipc_rank_topology(
    rank: int, device: Optional[torch.device] = None
) -> PcieIpcRankTopology:
    """Probe whether this rank's GPU shares a PCIe switch with any peer.

    Never raises: any failure is recorded in ``probe_error`` and the decision
    layer treats an unknown topology conservatively.

    Only the GPU this rank owns is probed against the other visible GPUs, so a
    job pinned to a subset of the machine describes that subset rather than the
    whole host. That matters on a mixed box where one island sits behind a
    switch and another does not.
    """
    topo = PcieIpcRankTopology(rank=rank)
    try:
        topo.hostname = socket.gethostname()
        parsed = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        if parsed.type != "cuda":
            raise ValueError(f"probe requires a CUDA device, got {parsed!r}")
        device_index = (
            parsed.index if parsed.index is not None else torch.cuda.current_device()
        )
        topo.device_index = device_index

        import pynvml

        pynvml.nvmlInit()
        try:

            def _uuid(idx: int) -> str:
                props = torch.cuda.get_device_properties(idx)
                uuid = getattr(props, "uuid", None)
                if uuid is None:
                    raise RuntimeError(
                        "torch.cuda.get_device_properties(...).uuid unavailable; "
                        "cannot establish physical GPU identity"
                    )
                return f"GPU-{uuid}"

            def _handle(idx: int):
                return pynvml.nvmlDeviceGetHandleByUUID(_uuid(idx).encode())

            topo.device_uuid = _uuid(device_index)
            my_handle = _handle(device_index)
            # NVML_TOPOLOGY_HOSTBRIDGE is the first level that leaves the switch
            # fabric, so anything strictly below it means the pair talks through
            # a PCIe switch without reaching the host bridge.
            hostbridge = pynvml.NVML_TOPOLOGY_HOSTBRIDGE
            for peer in range(torch.cuda.device_count()):
                if peer == device_index:
                    continue
                peer_uuid = _uuid(peer)
                try:
                    level = pynvml.nvmlDeviceGetTopologyCommonAncestor(
                        my_handle, _handle(peer)
                    )
                    topo.peer_switch_local[peer_uuid] = level < hostbridge
                except pynvml.NVMLError as pair_err:
                    topo.pair_errors[peer_uuid] = str(pair_err)
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001 - any probe failure => conservative fallback
        topo.probe_error = f"{type(e).__name__}: {e}"
    return topo


def decide_pcie_ipc_profile(
    requested: Optional[str], topologies: List[PcieIpcRankTopology]
) -> PcieIpcProfileDecision:
    """Pick the tuning table from the gathered probes. Pure function.

    An explicit ``requested`` profile always wins. Otherwise the group is
    switch-paired only if some rank positively observed a switch-local peer;
    anything unknown or unprobeable falls back to ``rootcplx-noswitch``,
    because misapplying the switch-paired table on a switch-free machine has
    been measured at up to 4.5x slower while the reverse only forgoes tuning.
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

    # Only pairs where BOTH endpoints belong to this group count. The probe
    # walks every GPU the process can see, which for a subgroup is a superset:
    # a switch-local pair outside the group says nothing about how the group's
    # own ranks talk to each other.
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
    rank = dist.get_rank(group=group)
    local = probe_pcie_ipc_rank_topology(rank, device=device)
    gathered: List[Optional[PcieIpcRankTopology]] = [None] * dist.get_world_size(
        group=group
    )
    dist.all_gather_object(gathered, local, group=group)
    return decide_pcie_ipc_profile(requested, [t for t in gathered if t is not None])
