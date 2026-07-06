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
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .ulysses_a2a import SUPPORTED_WORLD_SIZES

ULYSSES_BACKENDS = ("auto", "nvlink", "nccl")


class UlyssesBackendError(RuntimeError):
    """Raised when ``backend="nvlink"`` is forced but the topology cannot support it."""


@dataclass
class UlyssesRankTopology:
    """Per-rank topology probe result, exchanged across the group.

    ``peer_p2p`` / ``peer_nvlink`` are keyed by the *peer GPU's UUID* so the
    decision layer can join results across ranks regardless of each process's
    ``CUDA_VISIBLE_DEVICES`` ordering. A peer GPU that is not visible to this
    rank is simply absent from the dicts (treated as no-P2P).
    """

    rank: int
    hostname: str
    device_index: int
    device_uuid: str  # "GPU-xxxx..." or "" when unknown
    pci_bus_id: str
    peer_p2p: Dict[str, bool] = field(default_factory=dict)
    peer_nvlink: Dict[str, bool] = field(default_factory=dict)
    probe_error: Optional[str] = None


@dataclass(frozen=True)
class UlyssesBackendDecision:
    """Outcome of backend selection.

    ``backend`` is the backend actually chosen (``"nvlink"`` or ``"nccl"``);
    ``reason`` explains why — for NCCL it is the fallback reason, for NVLink it
    records what was verified.
    """

    backend: str
    reason: str


def probe_ulysses_rank_topology(device: torch.device, rank: int) -> UlyssesRankTopology:
    """Probe this rank's GPU identity and its P2P/NVLink reachability to every
    other CUDA device visible to this process.

    Never raises: any probing failure is recorded in ``probe_error`` so the
    (conservative) decision layer falls back to NCCL.
    """
    hostname = socket.gethostname()
    device_index = device.index if device.index is not None else 0
    topo = UlyssesRankTopology(
        rank=rank,
        hostname=hostname,
        device_index=device_index,
        device_uuid="",
        pci_bus_id="",
    )
    try:
        import pynvml

        pynvml.nvmlInit()
        try:

            def _uuid(idx: int) -> str:
                return f"GPU-{torch.cuda.get_device_properties(idx).uuid}"

            def _handle(idx: int):
                return pynvml.nvmlDeviceGetHandleByUUID(_uuid(idx).encode())

            topo.device_uuid = _uuid(device_index)
            my_handle = _handle(device_index)
            topo.pci_bus_id = pynvml.nvmlDeviceGetPciInfo(my_handle).busId
            if isinstance(topo.pci_bus_id, bytes):
                topo.pci_bus_id = topo.pci_bus_id.decode()

            for peer in range(torch.cuda.device_count()):
                if peer == device_index:
                    continue
                peer_uuid = _uuid(peer)
                topo.peer_p2p[peer_uuid] = torch.cuda.can_device_access_peer(
                    device_index, peer
                )
                # Pair-wise NVLink check: "this device has some active NVLink"
                # is NOT enough to prove a full mesh; ask NVML about the
                # concrete (my GPU, peer GPU) pair.
                try:
                    status = pynvml.nvmlDeviceGetP2PStatus(
                        my_handle, _handle(peer), pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                    topo.peer_nvlink[peer_uuid] = status == pynvml.NVML_P2P_STATUS_OK
                except pynvml.NVMLError:
                    topo.peer_nvlink[peer_uuid] = False
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001 — any probe failure => conservative fallback
        topo.probe_error = f"{type(e).__name__}: {e}"
    return topo


def decide_ulysses_backend(
    requested: str,
    topologies: List[UlyssesRankTopology],
    supported_world_sizes: Sequence[int] = SUPPORTED_WORLD_SIZES,
) -> UlyssesBackendDecision:
    """Pure decision function: gathered per-rank probes -> (backend, reason).

    Deterministic in its inputs, so every rank that gathers the same topology
    list computes the same decision. Conservative: anything unknown,
    inconsistent, or unverifiable selects NCCL. Raises
    :class:`UlyssesBackendError` only when ``requested == "nvlink"`` and the
    NVLink path cannot be used.
    """
    if requested not in ULYSSES_BACKENDS:
        raise ValueError(
            f"backend must be one of {ULYSSES_BACKENDS}, got {requested!r}"
        )

    def fallback(reason: str) -> UlyssesBackendDecision:
        if requested == "nvlink":
            raise UlyssesBackendError(f"backend='nvlink' requested but {reason}")
        return UlyssesBackendDecision("nccl", reason)

    if requested == "nccl":
        return UlyssesBackendDecision("nccl", "backend='nccl' requested")

    world_size = len(topologies)
    by_rank = sorted(topologies, key=lambda t: t.rank)
    if [t.rank for t in by_rank] != list(range(world_size)):
        return fallback(
            f"malformed topology info: ranks {[t.rank for t in topologies]} "
            f"are not exactly 0..{world_size - 1}"
        )

    for t in by_rank:
        if t.probe_error is not None:
            return fallback(f"topology probe failed on rank {t.rank}: {t.probe_error}")

    if world_size not in supported_world_sizes:
        return fallback(
            f"world size {world_size} not in fused-kernel supported sizes "
            f"{tuple(supported_world_sizes)}"
        )

    hostnames = {t.hostname for t in by_rank}
    if len(hostnames) > 1:
        return fallback(f"ranks span multiple hosts: {sorted(hostnames)}")

    for t in by_rank:
        if not t.device_uuid:
            return fallback(f"rank {t.rank} GPU identity unknown")

    uuid_to_rank: Dict[str, int] = {}
    for t in by_rank:
        if t.device_uuid in uuid_to_rank:
            return fallback(
                f"ranks {uuid_to_rank[t.device_uuid]} and {t.rank} share the same "
                f"physical GPU {t.device_uuid}"
            )
        uuid_to_rank[t.device_uuid] = t.rank

    for src in by_rank:
        for dst in by_rank:
            if src.rank == dst.rank:
                continue
            if not src.peer_p2p.get(dst.device_uuid, False):
                return fallback(
                    f"no P2P access from rank {src.rank} ({src.device_uuid}) to "
                    f"rank {dst.rank} ({dst.device_uuid})"
                )
            if not src.peer_nvlink.get(dst.device_uuid, False):
                return fallback(
                    f"no NVLink between rank {src.rank} ({src.device_uuid}) and "
                    f"rank {dst.rank} ({dst.device_uuid})"
                )

    return UlyssesBackendDecision(
        "nvlink",
        f"all-pairs NVLink P2P verified across {world_size} ranks on "
        f"{next(iter(hostnames))}",
    )


def resolve_ulysses_backend(
    backend: str = "auto",
    group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> UlyssesBackendDecision:
    """Group-consistent backend selection. Must run *before* any IPC allocation
    or JIT compilation; it performs no CUDA allocations itself.

    Every rank probes its local topology, the probes are all-gathered, and each
    rank evaluates the same pure decision function on the same gathered list.
    The resulting decisions are all-gathered again and cross-checked; any
    disagreement (which indicates non-deterministic probing, e.g. racing driver
    state) conservatively selects NCCL — or raises for ``backend="nvlink"``.
    """
    if backend not in ULYSSES_BACKENDS:
        raise ValueError(f"backend must be one of {ULYSSES_BACKENDS}, got {backend!r}")
    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    local = probe_ulysses_rank_topology(device, rank)

    topologies: List[Optional[UlyssesRankTopology]] = [None] * world_size
    dist.all_gather_object(topologies, local, group=group)

    # Deterministic in the gathered list, so on forced-NVLink failure every
    # rank raises here together — before any IPC allocation or JIT compile.
    decision = decide_ulysses_backend(backend, topologies)

    decisions: List[Optional[Tuple[str, str]]] = [None] * world_size
    dist.all_gather_object(decisions, (decision.backend, decision.reason), group=group)
    if any(d != decisions[0] for d in decisions):
        reason = f"inconsistent backend decisions across ranks: {decisions}"
        if backend == "nvlink":
            raise UlyssesBackendError(f"backend='nvlink' requested but {reason}")
        return UlyssesBackendDecision("nccl", reason)

    return decision
