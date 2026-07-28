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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

# world sizes for which the fused-transpose NVLink kernel is instantiated;
# lives here (the policy layer) so the dependency direction stays
# ulysses.py -> ulysses_topology.py with no cycle
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)

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
    hostname: str = ""
    device_index: int = -1
    device_uuid: str = ""  # "GPU-xxxx..." or "" when unknown
    pci_bus_id: str = ""
    peer_p2p: Dict[str, bool] = field(default_factory=dict)
    peer_nvlink: Dict[str, bool] = field(default_factory=dict)
    # peer uuid -> NVML error text for that concrete pair probe; distinguishes
    # "probe broke for this pair" from "pair verified to have no NVLink"
    pair_errors: Dict[str, str] = field(default_factory=dict)
    probe_error: Optional[str] = None


@dataclass(frozen=True)
class UlyssesBackendDecision:
    """Outcome of backend selection.

    ``backend`` is the backend actually chosen (``"nvlink"`` or ``"nccl"``);
    ``reason`` is the *selection* reason: for NVLink it records what was
    verified; for NCCL it is a fallback reason only under ``backend="auto"``
    (an explicit ``backend="nccl"`` request simply reports "requested").
    """

    backend: str
    reason: str


def probe_ulysses_rank_topology(
    device: Optional[Union[torch.device, str, int]], rank: int
) -> UlyssesRankTopology:
    """Probe this rank's GPU identity and its P2P/NVLink reachability to every
    other CUDA device visible to this process.

    Never raises: the whole probe (including hostname and device resolution)
    runs inside an exception envelope, so any failure lands in ``probe_error``
    and the (conservative) decision layer falls back to NCCL.
    """
    topo = UlyssesRankTopology(rank=rank)
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
       disagreement conservatively selects NCCL — or raises for
       ``backend="nvlink"``.
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
        local = probe_ulysses_rank_topology(device, rank)
    except Exception as e:  # noqa: BLE001
        local = UlyssesRankTopology(rank=rank, probe_error=f"{type(e).__name__}: {e}")

    topologies: List[Optional[UlyssesRankTopology]] = _guarded_gather(local)

    # ---- gather 3: decision outcomes (unconditional) ------------------------
    # ("ok", backend, reason) | ("backend_error", msg) | ("error", msg)
    outcome: Tuple[str, ...]
    try:
        decision = decide_ulysses_backend(requested, topologies)
        outcome = ("ok", decision.backend, decision.reason)
    except UlyssesBackendError as e:
        outcome = ("backend_error", str(e))
    except Exception as e:  # noqa: BLE001
        outcome = ("error", f"{type(e).__name__}: {e}")

    outcomes: List[Optional[Tuple[str, ...]]] = _guarded_gather(outcome)

    # Joint resolution: identical gathered list => identical result on every
    # rank, whether that is a raise or a decision. Invariants: only
    # requested="nvlink" may raise here (auto always degrades to NCCL, even if
    # a buggy decision layer raised UlyssesBackendError under auto), and
    # requested="nvlink" never silently returns anything but NVLink.
    backend_errors = [o for o in outcomes if o and o[0] == "backend_error"]
    if backend_errors:
        if requested == "nvlink":
            raise UlyssesBackendError(backend_errors[0][1])
        return UlyssesBackendDecision(
            "nccl", f"backend decision error: {backend_errors[0][1]}"
        )
    errors = {r: o[1] for r, o in enumerate(outcomes) if o and o[0] == "error"}
    if errors:
        reason = f"backend decision failed on rank(s) {errors}"
        if requested == "nvlink":
            raise UlyssesBackendError(f"backend='nvlink' requested but {reason}")
        return UlyssesBackendDecision("nccl", reason)
    if any(o != outcomes[0] for o in outcomes):
        reason = f"inconsistent backend decisions across ranks: {outcomes}"
        if requested == "nvlink":
            raise UlyssesBackendError(f"backend='nvlink' requested but {reason}")
        return UlyssesBackendDecision("nccl", reason)

    final = UlyssesBackendDecision(outcomes[0][1], outcomes[0][2])
    if requested == "nvlink" and final.backend != "nvlink":
        raise UlyssesBackendError(
            f"backend='nvlink' requested but the decision layer selected "
            f"{final.backend!r} ({final.reason}); refusing to silently violate "
            "the forced backend"
        )
    return final
