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

import ipaddress
import os
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

# world sizes for which the fused-transpose NVLink kernel is instantiated;
# lives here (the policy layer) so the dependency direction stays
# ulysses.py -> ulysses_topology.py with no cycle
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
PCIE_SUPPORTED_WORLD_SIZES = (1, 2, 4, 8)
# The 4+4 NUMA hybrid route (same-NUMA CUDA P2P plus cross-NUMA mlx5) exists
# only at this world size; every other supported size is pure CUDA P2P.
PCIE_HYBRID_WORLD_SIZE = 8
_PCIE_RDMA_PORT = 1
_PCIE_GID_INDICES_ENV = "FLASHINFER_ULYSSES_PCIE_GID_INDICES"

ULYSSES_BACKENDS = ("auto", "nvlink", "pcie", "nccl")


class UlyssesBackendError(RuntimeError):
    """Raised when a forced topology-specific backend cannot be supported."""


@dataclass
class UlyssesRankTopology:
    """Per-rank topology probe result, exchanged across the group.

    ``peer_p2p`` / ``peer_nvlink`` are keyed by the *peer GPU's UUID* so the
    decision layer can join results across ranks regardless of each process's
    ``CUDA_VISIBLE_DEVICES`` ordering. A peer GPU that is not visible to this
    rank is simply absent from the dicts (treated as no-P2P).
    """

    rank: int
    hostname: str = ""
    device_index: int = -1
    device_uuid: str = ""  # "GPU-xxxx..." or "" when unknown
    pci_bus_id: str = ""
    numa_node: int = -1
    nic_name: str = ""
    peer_p2p: Dict[str, bool] = field(default_factory=dict)
    peer_nvlink: Dict[str, bool] = field(default_factory=dict)
    # peer uuid -> NVML error text for that concrete pair probe; distinguishes
    # "probe broke for this pair" from "pair verified to have no NVLink"
    pair_errors: Dict[str, str] = field(default_factory=dict)
    # Raw FLASHINFER_ULYSSES_PCIE_ROUTE value; validated jointly by the
    # decision layer so every rank raises the same error on a bad setting.
    route: str = "auto"
    pcie_error: Optional[str] = None
    probe_error: Optional[str] = None
    # The selected rank-local GID index and the full explicit rank-ordered
    # override are gathered separately: the decision layer must detect two
    # ranks whose environments disagree.
    gid_index: int = -1
    gid_indices_override: Optional[Tuple[int, ...]] = None
    gid_error: Optional[str] = None


@dataclass(frozen=True)
class UlyssesPciePlan:
    """Group-wide route plan for the explicit PCIe backend.

    ``transport`` is ``"p2p"`` for an all-CUDA-copy route, ``"hybrid"`` for
    same-NUMA CUDA copies plus cross-NUMA mlx5, or ``"rdma"`` for per-rank
    mlx5 to every peer. Fields are indexed by process-group rank; they
    describe the sysfs plan and are runtime-verified by native
    initialization. A P2P plan has an empty GID tuple because it never opens
    an RDMA QP. ``requested_route`` records the joint
    ``FLASHINFER_ULYSSES_PCIE_ROUTE`` setting the plan was derived from.
    """

    numa_nodes: Tuple[int, ...]
    nic_names: Tuple[str, ...]
    transport: str = "hybrid"
    gid_indices: Tuple[int, ...] = ()
    requested_route: str = "auto"


@dataclass(frozen=True)
class UlyssesBackendDecision:
    """Outcome of backend selection.

    ``backend`` is the backend actually chosen (``"nvlink"``, ``"pcie"`` or
    ``"nccl"``);
    ``reason`` is the *selection* reason: for NVLink it records what was
    verified; for NCCL it is a fallback reason only under ``backend="auto"``
    (an explicit ``backend="nccl"`` request simply reports "requested").
    """

    backend: str
    reason: str
    pcie_plan: Optional[UlyssesPciePlan] = None


def _canonical_pci_path(pci_bus_id: str) -> Path:
    bus_id = pci_bus_id.lower()
    parts = bus_id.split(":")
    if len(parts) == 3 and len(parts[0]) > 4:
        bus_id = f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
    return (Path("/sys/bus/pci/devices") / bus_id).resolve(strict=True)


def _path_distance(left: Path, right: Path) -> int:
    common = 0
    for a, b in zip(left.parts, right.parts, strict=False):
        if a != b:
            break
        common += 1
    return len(left.parts) + len(right.parts) - 2 * common


def _parse_pcie_gid_indices_override() -> Optional[Tuple[int, ...]]:
    """Parse the rank-ordered GID override without weakening GID validation."""
    configured = os.getenv(_PCIE_GID_INDICES_ENV)
    if configured is None or configured == "":
        return None

    values = [value.strip() for value in configured.split(",")]
    if not values or any(re.fullmatch(r"[0-9]+", value) is None for value in values):
        raise ValueError(
            f"{_PCIE_GID_INDICES_ENV} must be a comma-separated list of "
            "non-negative decimal GID indices, one per rank in rank order"
        )
    indices = tuple(int(value, 10) for value in values)
    if any(index > 2**31 - 1 for index in indices):
        raise ValueError(f"{_PCIE_GID_INDICES_ENV} indices must fit in a C++ int")
    return indices


def _probe_rocev2_ipv4_gid(
    nic_name: str,
    requested_index: Optional[int] = None,
    *,
    sysfs_root: Path = Path("/sys"),
) -> int:
    """Pick the GID table index of an IPv4-mapped RoCE v2 entry on port 1.

    Automatic selection takes the lowest usable index; the rank-ordered
    override chooses among usable entries, and native initialization
    revalidates the index against the live GID table before creating QPs.
    """
    port_root = (
        sysfs_root / "class" / "infiniband" / nic_name / "ports" / str(_PCIE_RDMA_PORT)
    )
    candidates = []
    try:
        entries = sorted(
            (e for e in (port_root / "gids").iterdir() if e.name.isdigit()),
            key=lambda e: int(e.name, 10),
        )
    except OSError as err:
        raise RuntimeError(
            f"cannot enumerate GIDs for {nic_name} port {_PCIE_RDMA_PORT}: {err}"
        ) from err
    for entry in entries:
        try:
            address = ipaddress.IPv6Address(entry.read_text().strip())
            gid_type = (
                (port_root / "gid_attrs" / "types" / entry.name).read_text().strip()
            )
        except (OSError, ipaddress.AddressValueError):
            continue
        ipv4 = address.ipv4_mapped
        if gid_type == "RoCE v2" and ipv4 is not None and not ipv4.is_unspecified:
            candidates.append(int(entry.name, 10))
    if requested_index is not None:
        if requested_index in candidates:
            return requested_index
        raise RuntimeError(
            f"{_PCIE_GID_INDICES_ENV} selects index {requested_index} for "
            f"{nic_name} port {_PCIE_RDMA_PORT}, which is not a usable IPv4 "
            f"RoCE v2 entry (usable: {candidates or 'none'})"
        )
    if not candidates:
        raise RuntimeError(
            f"no usable IPv4 RoCE v2 GID found for {nic_name} port {_PCIE_RDMA_PORT}"
        )
    return candidates[0]


def _probe_pcie_route(pci_bus_id: str, rank: int) -> Tuple[int, str]:
    gpu_path = _canonical_pci_path(pci_bus_id)
    numa_node = int((gpu_path / "numa_node").read_text().strip())
    if numa_node < 0:
        raise RuntimeError(f"NUMA node is unknown for GPU {pci_bus_id}")

    configured = os.getenv("FLASHINFER_ULYSSES_PCIE_NICS")
    if configured:
        names = [name.strip() for name in configured.split(",")]
        if any(not name for name in names):
            raise ValueError(
                "FLASHINFER_ULYSSES_PCIE_NICS must be a comma-separated list "
                "of mlx5 device names, one per rank in rank order"
            )
        if rank >= len(names):
            raise ValueError(
                f"FLASHINFER_ULYSSES_PCIE_NICS has {len(names)} entries but "
                f"this is rank {rank}"
            )
        selected = names[rank]
        if not (Path("/sys/class/infiniband") / selected / "device").exists():
            raise RuntimeError(f"configured RDMA device {selected!r} does not exist")
        return numa_node, selected

    candidates = []
    for entry in Path("/sys/class/infiniband").glob("mlx5_*"):
        try:
            nic_path = (entry / "device").resolve(strict=True)
        except OSError:
            continue
        candidates.append((_path_distance(gpu_path, nic_path), entry.name))
    if not candidates:
        raise RuntimeError("no mlx5 RDMA devices found")
    candidates.sort(key=lambda item: (item[0], item[1]))
    minimum = candidates[0][0]
    closest = [name for distance, name in candidates if distance == minimum]
    closest.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    # Two NICs can tie for closest. Split adjacent GPUs across the tie by
    # physical PCI bus, not rank or CUDA ordinal, so the assignment survives
    # CUDA_VISIBLE_DEVICES reordering. FLASHINFER_ULYSSES_PCIE_NICS overrides.
    bus = int(gpu_path.name.split(":")[-2], 16)
    return numa_node, closest[bus % len(closest)]


def probe_ulysses_rank_topology(
    device: Optional[Union[torch.device, str, int]],
    rank: int,
    *,
    probe_pcie: bool = True,
) -> UlyssesRankTopology:
    """Probe this rank's GPU identity and its P2P/NVLink reachability to every
    other CUDA device visible to this process.

    Never raises: the whole probe (including hostname and device resolution)
    runs inside an exception envelope, so any failure lands in ``probe_error``
    and the (conservative) decision layer falls back to NCCL.

    Parameters
    ----------
    device : torch.device or str or int, optional
        CUDA device for this rank. ``None`` uses the current CUDA device.
    rank : int
        Rank id to record in the returned topology object.
    probe_pcie : bool
        Also walk sysfs for the NUMA node, mlx5 NIC and RoCE v2 GID this rank
        would use on the PCIe backend, filling ``numa_node``, ``nic_name``,
        ``gid_index`` and ``pcie_error``/``gid_error``. Only
        ``backend="pcie"`` consumes these; pass ``False`` to skip the walk.

    Returns
    -------
    UlyssesRankTopology
        Per-rank topology information, including any probe error instead of
        raising locally.
    """
    topo = UlyssesRankTopology(rank=rank)
    topo.route = os.environ.get("FLASHINFER_ULYSSES_PCIE_ROUTE", "") or "auto"
    try:
        topo.hostname = socket.gethostname()
        if device is None:
            parsed = torch.device("cuda", torch.cuda.current_device())
        else:
            parsed = torch.device(device)
        if parsed.type != "cuda":
            raise ValueError(
                f"Ulysses topology probe requires a CUDA device, got {parsed!r}"
            )
        # index=None means the *current* device, not GPU 0
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
                        "torch.cuda.get_device_properties(...).uuid unavailable "
                        "(torch too old); cannot establish physical GPU identity"
                    )
                return f"GPU-{uuid}"

            def _handle(idx: int):
                return pynvml.nvmlDeviceGetHandleByUUID(_uuid(idx).encode())

            topo.device_uuid = _uuid(device_index)
            my_handle = _handle(device_index)
            topo.pci_bus_id = pynvml.nvmlDeviceGetPciInfo(my_handle).busId
            if isinstance(topo.pci_bus_id, bytes):
                topo.pci_bus_id = topo.pci_bus_id.decode()

            # Only backend="pcie" consumes numa_node/nic_name, so skip the
            # sysfs walk otherwise. Its failure is recorded separately so a
            # host without verbs can still select NVLink.
            if probe_pcie:
                try:
                    topo.numa_node, topo.nic_name = _probe_pcie_route(
                        topo.pci_bus_id, rank
                    )
                except Exception as pcie_err:  # noqa: BLE001
                    topo.pcie_error = f"{type(pcie_err).__name__}: {pcie_err}"
                if topo.pcie_error is None and topo.route != "p2p":
                    try:
                        override = _parse_pcie_gid_indices_override()
                        topo.gid_indices_override = override
                        if override is not None and not 0 <= rank < len(override):
                            raise ValueError(
                                f"rank {rank} is outside the {_PCIE_GID_INDICES_ENV} "
                                "rank-ordered list"
                            )
                        topo.gid_index = _probe_rocev2_ipv4_gid(
                            topo.nic_name,
                            None if override is None else override[rank],
                        )
                    except Exception as gid_err:  # noqa: BLE001
                        topo.gid_error = f"{type(gid_err).__name__}: {gid_err}"

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
                except pynvml.NVMLError as pair_err:
                    # record the diagnostic instead of silently pretending the
                    # physical link is absent
                    topo.peer_nvlink[peer_uuid] = False
                    topo.pair_errors[peer_uuid] = str(pair_err)
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001 — any probe failure => conservative fallback
        topo.probe_error = f"{type(e).__name__}: {e}"
    return topo


def _rdma_route_error(
    by_rank: List["UlyssesRankTopology"],
    gid_indices: Tuple[int, ...],
    all_p2p_error: Optional[str],
    *,
    require_numa_split: bool,
) -> Optional[str]:
    """Why an RDMA-carrying route (hybrid or all-RDMA) cannot be planned.

    Check order matters: an operator who lacks full-group P2P must not be
    shown a GID error.
    """
    if all_p2p_error is not None:
        return (
            "an RDMA route still requires full-group CUDA P2P access for its "
            f"epoch signal barrier: {all_p2p_error}"
        )
    for t in by_rank:
        if t.pcie_error is not None:
            return f"PCIe topology probe failed on rank {t.rank}: {t.pcie_error}"
        if t.numa_node < 0 or not t.nic_name:
            return (
                f"PCIe route is incomplete on rank {t.rank}: "
                f"numa={t.numa_node}, nic={t.nic_name!r}"
            )
        if t.gid_error is not None:
            return f"RoCE GID probe failed on rank {t.rank}: {t.gid_error}"
        if t.gid_index < 0:
            return f"no RoCE GID was selected on rank {t.rank}"
    overrides = [t.gid_indices_override for t in by_rank]
    if any(override != overrides[0] for override in overrides[1:]):
        return f"{_PCIE_GID_INDICES_ENV} is inconsistent across ranks: {overrides}"
    if overrides[0] is not None and gid_indices != overrides[0][: len(gid_indices)]:
        return (
            f"selected GID indices {gid_indices} do not match "
            f"{_PCIE_GID_INDICES_ENV}={overrides[0]}"
        )
    # Per-rank NIC selection is independent, so two GPUs can land on the same
    # device; only the full rank-ordered plan can detect it.
    assigned = [t.nic_name for t in by_rank]
    if len(set(assigned)) != len(assigned):
        return (
            f"one mlx5 device is assigned to more than one rank ({assigned}); "
            "an RDMA route needs a distinct NIC per rank — fix "
            "FLASHINFER_ULYSSES_PCIE_NICS if set, or set it to a rank-ordered "
            "list to override automatic routing"
        )
    if require_numa_split:
        counts: Dict[int, int] = {}
        for t in by_rank:
            counts[t.numa_node] = counts.get(t.numa_node, 0) + 1
        half = PCIE_HYBRID_WORLD_SIZE // 2
        if sorted(counts.values()) != [half, half]:
            return f"hybrid transport requires a 4+4 NUMA split, got {counts}"
    return None


def decide_ulysses_backend(
    requested: str,
    topologies: List[UlyssesRankTopology],
    supported_world_sizes: Sequence[int] = SUPPORTED_WORLD_SIZES,
) -> UlyssesBackendDecision:
    """Pure decision function: gathered per-rank probes -> (backend, reason).

    Deterministic in its inputs, so every rank that gathers the same topology
    list computes the same decision. Conservative: anything unknown,
    inconsistent, or unverifiable selects NCCL. Raises
    :class:`UlyssesBackendError` only when ``"nvlink"`` or ``"pcie"`` is
    explicitly requested and that path cannot be used.

    Parameters
    ----------
    requested : str
        Requested backend policy: ``"auto"``, ``"nvlink"``, ``"pcie"``, or
        ``"nccl"``.
    topologies : list[UlyssesRankTopology]
        Per-rank topology probe results gathered from the process group.
    supported_world_sizes : Sequence[int], optional
        World sizes for which the fused NVLink kernel is instantiated.

    Returns
    -------
    UlyssesBackendDecision
        Backend choice and selection or fallback reason.
    """
    if requested not in ULYSSES_BACKENDS:
        raise ValueError(
            f"backend must be one of {ULYSSES_BACKENDS}, got {requested!r}"
        )

    def fallback(reason: str) -> UlyssesBackendDecision:
        if requested in ("nvlink", "pcie"):
            raise UlyssesBackendError(f"backend={requested!r} requested but {reason}")
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

    if requested == "pcie" and world_size not in PCIE_SUPPORTED_WORLD_SIZES:
        return fallback(
            f"PCIe backend supports world sizes {PCIE_SUPPORTED_WORLD_SIZES}, "
            f"got {world_size}"
        )

    if requested != "pcie" and world_size not in supported_world_sizes:
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

    if requested == "pcie":
        numa_nodes = tuple(t.numa_node for t in by_rank)
        nic_names = tuple(t.nic_name for t in by_rank)
        gid_indices = tuple(t.gid_index for t in by_rank)
        routes = {t.route for t in by_rank}
        if len(routes) > 1:
            return fallback(
                "ranks disagree on FLASHINFER_ULYSSES_PCIE_ROUTE: "
                f"{sorted(routes)}; every rank must set it identically"
            )
        route = routes.pop()
        if route not in ("auto", "p2p", "rdma", "hybrid"):
            return fallback(
                f"invalid FLASHINFER_ULYSSES_PCIE_ROUTE {route!r}: "
                "use auto, p2p, rdma or hybrid"
            )

        all_p2p_error = None
        for src in by_rank:
            for dst in by_rank:
                if src.rank == dst.rank:
                    continue
                if not src.peer_p2p.get(dst.device_uuid, False):
                    all_p2p_error = (
                        f"no CUDA P2P access from rank {src.rank} "
                        f"({src.device_uuid}) to rank {dst.rank} ({dst.device_uuid})"
                    )
                    break
            if all_p2p_error is not None:
                break

        def p2p_plan(reason: str) -> UlyssesBackendDecision:
            return UlyssesBackendDecision(
                "pcie",
                reason,
                UlyssesPciePlan(
                    numa_nodes=numa_nodes,
                    nic_names=nic_names,
                    transport="p2p",
                    requested_route=route,
                ),
            )

        if world_size == 1:
            # One rank is an identity path; no route carries any payload.
            return p2p_plan("single-node 1-rank CUDA P2P route planned")

        if route == "rdma" or (
            route == "auto" and world_size == PCIE_HYBRID_WORLD_SIZE
        ):
            rdma_error = _rdma_route_error(
                by_rank, gid_indices, all_p2p_error, require_numa_split=False
            )
            forced = " (FLASHINFER_ULYSSES_PCIE_ROUTE=rdma)" if route == "rdma" else ""
            if rdma_error is None:
                return UlyssesBackendDecision(
                    "pcie",
                    f"single-node {world_size}-rank all-RDMA route planned: "
                    f"per-rank mlx5 RoCE to every peer{forced}",
                    UlyssesPciePlan(
                        numa_nodes=numa_nodes,
                        nic_names=nic_names,
                        transport="rdma",
                        gid_indices=gid_indices,
                        requested_route=route,
                    ),
                )
            if all_p2p_error is None:
                return p2p_plan(
                    f"single-node {world_size}-rank CUDA P2P route planned; "
                    f"all-RDMA unavailable: {rdma_error}"
                )
            return fallback(
                "neither all-RDMA nor all-P2P transport is available: "
                f"{rdma_error}; {all_p2p_error}"
            )

        if route == "hybrid":
            hybrid_error = _rdma_route_error(
                by_rank, gid_indices, all_p2p_error, require_numa_split=True
            )
            if hybrid_error is None:
                return UlyssesBackendDecision(
                    "pcie",
                    "single-node 4+4 NUMA hybrid route planned: same-NUMA CUDA "
                    "P2P plus cross-NUMA mlx5 RoCE "
                    "(FLASHINFER_ULYSSES_PCIE_ROUTE=hybrid)",
                    UlyssesPciePlan(
                        numa_nodes=numa_nodes,
                        nic_names=nic_names,
                        transport="hybrid",
                        gid_indices=gid_indices,
                        requested_route=route,
                    ),
                )
            if all_p2p_error is None:
                return p2p_plan(
                    f"single-node {world_size}-rank CUDA P2P route planned; "
                    f"hybrid unavailable: {hybrid_error}"
                )
            return fallback(
                f"neither hybrid nor all-P2P transport is available: "
                f"{hybrid_error}; {all_p2p_error}"
            )

        if all_p2p_error is not None:
            return fallback(all_p2p_error)
        forced = " (FLASHINFER_ULYSSES_PCIE_ROUTE=p2p)" if route == "p2p" else ""
        return p2p_plan(f"single-node {world_size}-rank CUDA P2P route planned{forced}")

    for src in by_rank:
        for dst in by_rank:
            if src.rank == dst.rank:
                continue
            if not src.peer_p2p.get(dst.device_uuid, False):
                return fallback(
                    f"no P2P access from rank {src.rank} ({src.device_uuid}) to "
                    f"rank {dst.rank} ({dst.device_uuid})"
                )
            if dst.device_uuid in src.pair_errors:
                return fallback(
                    f"NVLink probe failed between rank {src.rank} and rank "
                    f"{dst.rank}: {src.pair_errors[dst.device_uuid]}"
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
    device: Optional[Union[torch.device, str, int]] = None,
) -> UlyssesBackendDecision:
    """Group-consistent backend selection. Must run *before* any IPC allocation
    or JIT compilation. It allocates no IPC workspace and compiles nothing;
    the ``all_gather_object`` metadata collectives may themselves stage
    through CUDA buffers on NCCL process groups.

    Collective-safe outcome protocol: every rank participates in the same
    fixed *prefix* of ``all_gather_object`` calls (at most three; a group-wide
    explicit NCCL request or an invalid/inconsistent request exits jointly
    after the first) no matter what fails locally —
    rank-local errors are encoded as serializable outcomes and re-raised (or
    turned into an NCCL fallback) *jointly* after the gather, so no rank can
    leave the collective sequence early and deadlock its peers. The only
    uncoordinated failure mode left is the process group itself failing.

    Sequence:

    1. gather every rank's *requested* backend; jointly reject invalid or
       inconsistent requests. A group-wide explicit ``"nccl"`` request returns
       here, skipping the CUDA/NVML topology probe entirely (the gather
       itself may stage through CUDA on NCCL groups).
    2. gather every rank's probe outcome (the probe never raises; even a buggy
       probe implementation is caught into the outcome).
    3. every rank evaluates the same pure decision on the same gathered list,
       catches the result into an outcome, gathers, and cross-checks. Any
       disagreement conservatively selects NCCL — or raises for an explicitly
       forced ``backend="nvlink"`` / ``backend="pcie"``.

    Parameters
    ----------
    backend : str, default = "auto"
        Requested backend policy: ``"auto"``, ``"nvlink"``, ``"pcie"``, or
        ``"nccl"``.
    group : torch.distributed.ProcessGroup, optional
        Process group whose ranks must make a consistent backend decision.
        Defaults to ``torch.distributed.group.WORLD``.
    device : torch.device or str or int, optional
        CUDA device to use for probe and metadata collective guards.

    Returns
    -------
    UlyssesBackendDecision
        Group-consistent backend choice and selection or fallback reason.
    """
    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    # Metadata collectives must run bound to the caller's device: NCCL object
    # collectives stage through a tensor on the *current* device, so without a
    # guard every rank could land on GPU 0 when the caller relies on device=
    # instead of set_device. Any guard failure is a probe-level concern; keep
    # the guard best-effort and never raise before a gather.
    def _guarded_gather(payload: Any) -> List[Any]:
        out: List[Any] = [None] * world_size
        guard_index: Optional[int] = None
        try:
            if device is not None:
                parsed = torch.device(device)
                if parsed.type == "cuda" and parsed.index is not None:
                    guard_index = parsed.index
        except Exception:  # noqa: BLE001 — invalid device surfaces via probe
            guard_index = None
        if guard_index is not None:
            with torch.cuda.device(guard_index):
                dist.all_gather_object(out, payload, group=group)
        else:
            dist.all_gather_object(out, payload, group=group)
        return out

    # ---- gather 1: requested backends (before any local validation) --------
    # No user code may run before the gather: str(backend) would invoke a user
    # __str__ that could raise on one rank and hang the peers. Exact type check
    # (not isinstance) also excludes str subclasses with custom behavior; only
    # the interpreter-provided type name is used for invalid payloads.
    if type(backend) is str:
        request_payload = backend
    else:
        request_payload = f"<invalid type: {type(backend).__name__}>"
    requests: List[Optional[str]] = _guarded_gather(request_payload)

    invalid = {r: req for r, req in enumerate(requests) if req not in ULYSSES_BACKENDS}
    if invalid:
        # every rank raises the same error together
        raise ValueError(
            f"backend must be one of {ULYSSES_BACKENDS}; invalid request(s) "
            f"by rank: {invalid}"
        )
    if len(set(requests)) > 1:
        raise ValueError(
            f"inconsistent backend requests across ranks: {requests}; all ranks "
            "must pass the same backend"
        )
    requested = requests[0]
    if requested == "nccl":
        return UlyssesBackendDecision("nccl", "backend='nccl' requested")

    # ---- gather 2: probe outcomes ------------------------------------------
    # probe_ulysses_rank_topology never raises by contract, but a buggy or
    # monkeypatched probe must not break the collective sequence either.
    try:
        local = probe_ulysses_rank_topology(
            device, rank, probe_pcie=requested == "pcie"
        )
    except Exception as e:  # noqa: BLE001
        local = UlyssesRankTopology(rank=rank, probe_error=f"{type(e).__name__}: {e}")

    topologies: List[Optional[UlyssesRankTopology]] = _guarded_gather(local)

    # ---- gather 3: decision outcomes (unconditional) ------------------------
    # ("ok", backend, reason, pcie_plan) | ("backend_error", msg) | ("error", msg)
    # The payload is heterogeneous: the ok-branch carries an optional
    # UlyssesPciePlan alongside the two strings.
    outcome: Tuple[Any, ...]
    try:
        decision = decide_ulysses_backend(requested, topologies)
        outcome = ("ok", decision.backend, decision.reason, decision.pcie_plan)
    except UlyssesBackendError as e:
        outcome = ("backend_error", str(e))
    except Exception as e:  # noqa: BLE001
        outcome = ("error", f"{type(e).__name__}: {e}")

    outcomes: List[Optional[Tuple[Any, ...]]] = _guarded_gather(outcome)

    # Joint resolution: identical gathered list => identical result on every
    # rank, whether that is a raise or a decision. Invariants: only
    # a forced backend may raise here (auto always degrades to NCCL, even if
    # a buggy decision layer raised UlyssesBackendError under auto), and
    # requested="nvlink" never silently returns anything but NVLink.
    backend_errors = [o for o in outcomes if o and o[0] == "backend_error"]
    if backend_errors:
        if requested in ("nvlink", "pcie"):
            raise UlyssesBackendError(backend_errors[0][1])
        return UlyssesBackendDecision(
            "nccl", f"backend decision error: {backend_errors[0][1]}"
        )
    errors = {r: o[1] for r, o in enumerate(outcomes) if o and o[0] == "error"}
    if errors:
        reason = f"backend decision failed on rank(s) {errors}"
        if requested in ("nvlink", "pcie"):
            raise UlyssesBackendError(f"backend={requested!r} requested but {reason}")
        return UlyssesBackendDecision("nccl", reason)
    if any(o != outcomes[0] for o in outcomes):
        reason = f"inconsistent backend decisions across ranks: {outcomes}"
        if requested in ("nvlink", "pcie"):
            raise UlyssesBackendError(f"backend={requested!r} requested but {reason}")
        return UlyssesBackendDecision("nccl", reason)

    final = UlyssesBackendDecision(outcomes[0][1], outcomes[0][2], outcomes[0][3])
    if requested in ("nvlink", "pcie") and final.backend != requested:
        raise UlyssesBackendError(
            f"backend={requested!r} requested but the decision layer selected "
            f"{final.backend!r} ({final.reason}); refusing to silently violate "
            "the forced backend"
        )
    return final
