# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host contracts for the legacy NVFP4 GIN migration.

The live adapter owns a single registered arena and uses the borrowed default
process group to agree on errors before resource and execution boundaries.
Importing this module needs neither Torch nor NCCL; device dependencies are lazy.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable
from weakref import WeakKeyDictionary


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _align(value: int) -> int:
    return (value + 127) // 128 * 128


@dataclass(frozen=True)
class GinArenaRegion:
    """A byte range relative to the one registered root allocation."""

    name: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class GinArenaLayout:
    """Immutable arena partition, identical on every rank.

    ``gin_transport`` starts at zero because device GIN offsets are relative
    to the registered window. Weights, local workspace and final separate
    TopkReduce output need no LSA mapping and stay outside this allocation.
    """

    regions: tuple[GinArenaRegion, ...]
    total_bytes: int

    def region(self, name: str) -> GinArenaRegion:
        """Return the named range; raise KeyError if it is absent."""
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)

    def validate(self) -> None:
        """Reject overlaps, duplicate names and an invalid transport origin."""
        _positive_int("total_bytes", self.total_bytes)
        if self.total_bytes % 128:
            raise ValueError("arena total_bytes must be 128-byte aligned.")
        if not self.regions or self.regions[0].name != "gin_transport":
            raise ValueError("gin_transport must be the first arena region.")
        if self.regions[0].offset != 0:
            raise ValueError("gin_transport must start at registered-window offset 0.")
        end = 0
        names = set()
        for region in self.regions:
            _positive_int(f"{region.name}.nbytes", region.nbytes)
            if (
                not region.name
                or region.name in names
                or type(region.offset) is not int
                or region.offset % 128
                or region.offset < end
            ):
                raise ValueError(f"invalid or overlapping arena region: {region!r}")
            names.add(region.name)
            end = region.offset + region.nbytes
        if end > self.total_bytes:
            raise ValueError("arena region exceeds the registered root allocation.")


def build_gin_arena_layout(
    *,
    transport_bytes: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    shared_workspace_bytes: int,
) -> GinArenaLayout:
    """Partition NVFP4 inputs and shared workspace after the GIN transport.

    The caller obtains ``transport_bytes`` from the donor window layout and
    ``shared_workspace_bytes`` from the compiled kernel, without guessing
    device workspace offsets. SF rows use the legacy plain FP8 byte layout.
    """
    for name, value in (
        ("transport_bytes", transport_bytes),
        ("num_tokens", num_tokens),
        ("hidden", hidden),
        ("num_topk", num_topk),
        ("shared_workspace_bytes", shared_workspace_bytes),
    ):
        _positive_int(name, value)
    if hidden % 16:
        raise ValueError("NVFP4 hidden must be divisible by the 16-element SF block.")
    sizes = (
        ("gin_transport", transport_bytes),
        ("activation", num_tokens * (hidden // 2)),
        ("activation_sf", num_tokens * ((hidden // 16 + 3) // 4 * 4)),
        ("topk_weights", num_tokens * num_topk * 4),
        ("shared_workspace", shared_workspace_bytes),
    )
    regions = []
    cursor = 0
    for name, nbytes in sizes:
        cursor = _align(cursor)
        regions.append(GinArenaRegion(name, cursor, nbytes))
        cursor += nbytes
    layout = GinArenaLayout(tuple(regions), _align(cursor))
    layout.validate()
    return layout


def gin_arena_view(root, region: GinArenaRegion, *, dtype, shape: tuple[int, ...]):
    """Carve a typed view from a contiguous root uint8 CUDA tensor.

    Every peer-visible tensor must be carved here; independently allocated
    tensors cannot share this window's peer-pointer deltas. The returned view
    shares storage with ``root``. Torch is imported only on this allocation path.
    """
    import math

    import torch

    if (
        not root.is_cuda
        or root.dtype != torch.uint8
        or root.ndim != 1
        or not root.is_contiguous()
        or root.storage_offset() != 0
        or root.untyped_storage().nbytes() != root.numel()
    ):
        raise ValueError(
            "GIN arena must be an entire contiguous uint8 CUDA allocation."
        )
    if any(type(extent) is not int or extent <= 0 for extent in shape):
        raise ValueError("GIN arena view extents must be positive integers.")
    element_bytes = torch.empty((), dtype=dtype).element_size()
    if math.prod(shape) * element_bytes != region.nbytes:
        raise ValueError("GIN arena view shape/dtype does not match its byte region.")
    if (
        region.offset < 0
        or region.offset % 128
        or region.offset + region.nbytes > root.numel()
    ):
        raise ValueError("GIN arena view is outside the root or is not aligned.")
    return root.narrow(0, region.offset, region.nbytes).view(dtype).reshape(shape)


def lsa_peer_offsets(
    *,
    rank: int,
    world_size: int,
    ranks_per_node: int,
    base_address: int,
    peer_addresses: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    """Validate same-domain root addresses before constructing the DSL mapper.

    NCCL window peer queries must supply the contiguous same-domain global
    ranks ``rank // R * R .. rank // R * R + R`` in that order. Cross-domain
    peer IDs and absent addresses are rejected. The live adapter must obtain
    these addresses from this root's registered window, never another tensor.
    """
    _positive_int("world_size", world_size)
    _positive_int("ranks_per_node", ranks_per_node)
    _positive_int("base_address", base_address)
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size).")
    if world_size % ranks_per_node or ranks_per_node > 16:
        raise ValueError("LSA mapping requires R <= 16 dividing world_size.")
    expected = tuple(
        range(
            rank // ranks_per_node * ranks_per_node,
            (rank // ranks_per_node + 1) * ranks_per_node,
        )
    )
    if tuple(peer for peer, _ in peer_addresses) != expected:
        raise ValueError(f"LSA peer addresses must cover same-domain ranks {expected}.")
    for peer, address in peer_addresses:
        _positive_int(f"peer {peer} address", address)
        if address % 128:
            raise ValueError("LSA root addresses must be 128-byte aligned.")
    # A registered window may expose another VA alias even for self. NCCL
    # promises the peer's window storage, not identity with the input VA.
    # Preserve its delta exactly; the device peer probe verifies alias data.
    return tuple(address - base_address for _, address in peer_addresses)


class GinPreflightError(RuntimeError):
    """All participating default-group ranks rejected a runtime boundary."""


def require_gin_apis() -> None:
    """Check the donor's required nccl4py API surface without creating a comm.

    Catch this exception into the local preflight report, then gather errors
    on the existing default group before creating any NCCL resources. API
    availability alone does not establish usable RDMA or a valid LSA domain.
    """
    try:
        from nccl.core import (
            Communicator,
            NCCLDevCommRequirements,
            NcclGinConnectionType,
            UniqueId,
            WindowFlag,
            get_unique_id,
        )
        from nccl.core.device.cute import Gin
        from nccl.core.interop.torch import empty
    except Exception as exc:
        raise GinPreflightError(f"NCCL GIN dependencies unavailable: {exc}") from exc
    # These host APIs are needed in addition to the device signal protocol.
    required = (
        (
            Gin,
            (
                "put",
                "signal",
                "wait_signal",
                "read_signal",
                "reset_signal",
                "signal_shadow_pointer",
            ),
        ),
        (Communicator, ("init", "create_dev_comm", "register_window")),
        (NcclGinConnectionType, ("RAIL",)),
        (UniqueId, ("from_bytes",)),
        (WindowFlag, ("COLL_SYMMETRIC",)),
    )
    missing = [
        f"{obj.__name__}.{name}"
        for obj, names in required
        for name in names
        if not hasattr(obj, name)
    ]
    if not all(callable(fn) for fn in (NCCLDevCommRequirements, get_unique_id, empty)):
        missing.append("NCCL host constructors/allocator")
    if missing:
        raise GinPreflightError("NCCL GIN APIs missing: " + ", ".join(missing))


def validate_gin_rail_support(comm, *, world_size: int, ranks_per_node: int) -> None:
    """Reject unsupported RDMA/device APIs and a collapsed logical topology.

    Call after communicator creation, but before creating the device comm or
    registering the arena, and aggregate any exception on the control group.
    The LSA peer-pointer probe must additionally verify every local peer.
    """
    _positive_int("world_size", world_size)
    _positive_int("ranks_per_node", ranks_per_node)
    if world_size % ranks_per_node or world_size // ranks_per_node < 2:
        raise GinPreflightError("GIN requires at least two equal NVLink domains.")
    if ranks_per_node > 16:
        raise GinPreflightError("LSA mapper R > 16 has no validated launch ABI.")
    if not comm.device_api_support:
        raise GinPreflightError("NCCL communicator reports device_api_support=False.")
    gin_type = comm.railed_gin_type
    if (
        gin_type is None
        or str(getattr(gin_type, "name", gin_type)).split(".")[-1] == "NONE"
    ):
        raise GinPreflightError(
            "NCCL communicator reports railed GIN/RDMA unavailable."
        )
    expected = world_size // ranks_per_node
    if comm.n_lsa_teams != expected:
        raise GinPreflightError(
            f"NCCL rail topology has {comm.n_lsa_teams} LSA teams; expected {expected}."
        )


_gin_control_groups: Any = WeakKeyDictionary()


def _gin_control_group(dist):
    """Reuse CPU WORLD or initialize one CPU group before any graph capture.

    GIN sessions are initialized collectively outside capture. A NCCL-only
    default group needs a Gloo control group so capture-time errors never
    enqueue CUDA collectives. The default PG owns the child group's lifetime;
    destroying WORLD tears it down along with the other process groups.
    """
    world = dist.group.WORLD
    get_config = getattr(dist, "get_backend_config", None)
    if get_config is None:
        return world, False
    if "cpu:gloo" in str(get_config(world)):
        return world, True
    group = _gin_control_groups.get(world)
    if group is None:
        group = dist.new_group(backend="gloo")
        _gin_control_groups[world] = group
    return group, True


def collective_gin_preflight(
    dist,
    *,
    signature: object,
    local_error: str | None,
    phase: str,
    requested_group=None,
) -> None:
    """Agree on rank-local errors and configuration before a GIN boundary.

    All default-group ranks must call this in the same order, including ranks
    outside a requested subgroup. A subgroup is rejected through the default
    group so its members never enter a global UID broadcast alone. The caller
    captures local validation/import/allocation failures in ``local_error``.
    This function creates no communicator and registers no window.

    An unavailable control group or collective failure is job-fatal: callers
    must rely on the launcher's whole-job timeout/exit, never proceed to GIN.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise GinPreflightError("GIN requires the initialized default process group.")
    world = dist.group.WORLD
    if requested_group is not None and requested_group is not world:
        local_error = "; ".join(
            filter(None, (local_error, "EP subgroups are unsupported."))
        )
    control, cpu_control = _gin_control_group(dist)
    if cpu_control:
        import hashlib
        import torch

        # Use the same fixed-size CPU header at EVERY boundary, including
        # initialization and failures. Mixing compact and object protocols by
        # phase would hang when different ranks enter different boundaries.
        # CPU control also remains usable when a rank's CUDA binding is wrong.
        # A mismatch takes the original detailed-report path on every rank.
        fingerprint = hashlib.blake2b(
            repr((phase, signature)).encode(), digest_size=16
        ).digest()
        words = tuple(
            int.from_bytes(fingerprint[offset : offset + 8], "little", signed=True)
            for offset in (0, 8)
        )
        rank = dist.get_rank(group=world)
        size = dist.get_world_size(group=world)
        header = torch.tensor(
            (rank, *words, bool(local_error)), dtype=torch.int64, device="cpu"
        )
        headers = torch.empty(size * 4, dtype=torch.int64, device="cpu")
        dist.all_gather_into_tensor(headers, header, group=control)
        if all(
            entry[0] == peer and tuple(entry[1:3]) == words and not entry[3]
            for peer, entry in enumerate(headers.view(size, 4).tolist())
        ):
            return
    report = (dist.get_rank(group=world), phase, signature, local_error)
    reports: list[Any] = [None] * dist.get_world_size(group=world)
    dist.all_gather_object(reports, report, group=control)
    errors = []
    baseline = reports[0]
    baseline_valid = isinstance(baseline, tuple) and len(baseline) == 4
    for expected_rank, entry in enumerate(reports):
        if not isinstance(entry, tuple) or len(entry) != 4:
            errors.append(f"rank {expected_rank}: malformed preflight report")
            continue
        rank, other_phase, other_signature, error = entry
        if rank != expected_rank or other_phase != phase:
            errors.append(f"rank {expected_rank}: preflight phase/rank mismatch")
        if baseline_valid and other_signature != baseline[2]:
            errors.append(f"rank {expected_rank}: configuration differs from rank 0")
        if error:
            errors.append(f"rank {expected_rank}: {error}")
    if errors:
        raise GinPreflightError(f"GIN {phase} rejected: " + "; ".join(errors))


class GinResourceOwner:
    """Own a root arena and its window/device communicator/communicator.

    This is the lifecycle primitive used by the live NCCL factory below.
    ``synchronize`` must drain
    all kernels using these handles. Borrowed process groups and inputs are
    never attached here. Buffer views must not escape the owner's lifetime.
    """

    def __init__(self, *, synchronize: Callable[[], None]):
        self._synchronize = synchronize
        self.root = None
        self.window = None
        self.dev_comm = None
        self.comm = None
        self._closed = False

    def close(self) -> None:
        """Drain kernels, then release dependencies in reverse order.

        Stop on cleanup failure and keep dependent objects alive. Retrying
        ``close`` resumes at the failed resource; already released handles
        are never destroyed twice. A failed drain releases nothing.
        """
        if self._closed:
            return
        self._synchronize()
        for name in ("window", "dev_comm", "comm"):
            resource = getattr(self, name)
            if resource is not None:
                close = getattr(resource, "close", None)
                if close is None:
                    close = resource.destroy
                close()
                setattr(self, name, None)
        self.root = None
        self._closed = True


class GinFatalError(RuntimeError):
    """A resource/control collective failed; the entire job must terminate."""


# A failed NCCL collective can leave an unknown subset of ranks inside NCCL.
# Keep the dependencies alive until process exit instead of attempting another
# collective in a destructor, freeing registered storage or reusing the comm.
_failed_sessions: list[GinSession] = []


class GinSession(GinResourceOwner):
    """Own one GIN communicator, registered root and its local peer mappings."""

    def __init__(self, *, device: int, layout: GinArenaLayout):
        import torch

        super().__init__(synchronize=lambda: torch.cuda.synchronize(device))
        self.device = device
        self.layout = layout
        self.peer_offsets = ()
        self.identity: str | None = None
        self.views: dict[str, Any] = {}

    def view(self, name: str, *, dtype, shape: tuple[int, ...]):
        """Return a typed arena region and retain it for the session lifetime."""
        if self._closed or self.root is None:
            raise RuntimeError("GIN session has no live registered root.")
        result = gin_arena_view(
            self.root, self.layout.region(name), dtype=dtype, shape=shape
        )
        self.views[name] = result
        return result

    @contextmanager
    def workspace_construction(self):
        """Publish a workspace only after all ranks finish local allocations."""
        import torch.distributed as dist

        error = None
        try:
            yield
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            collective_gin_preflight(
                dist,
                signature=self.identity,
                local_error=error,
                phase="workspace-allocation",
            )
        except GinPreflightError:
            cleanup_error = None
            try:
                self.close()
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
                _failed_sessions.append(self)
            try:
                collective_gin_preflight(
                    dist,
                    signature=self.identity,
                    local_error=cleanup_error,
                    phase="workspace-cleanup",
                )
            except Exception as exc:
                raise GinFatalError(
                    "GIN workspace cleanup failed; terminate all ranks."
                ) from exc
            raise
        except Exception as exc:
            # A broken control plane cannot establish safe collective teardown.
            _failed_sessions.append(self)
            raise GinFatalError(
                "GIN workspace agreement failed; terminate all ranks."
            ) from exc

    def close(self) -> None:
        super().close()
        self.views.clear()


def create_gin_session(
    *,
    rank: int,
    world_size: int,
    ranks_per_node: int,
    device: int,
    layout: GinArenaLayout,
    signal_count: int,
    signature: tuple,
    process_group=None,
) -> GinSession:
    """Collectively create one runtime on an existing, correctly bound PG.

    All default-group ranks must enter, even for a rejected subgroup request.
    Local validation/allocation errors are gathered before the next resource
    boundary. No support-check bypass or forced GIN mode is offered. Failed
    NCCL/control collectives are job-fatal, with resources retained until exit.
    The caller must wrap the complete distributed job in a timeout.
    """
    import torch
    import torch.distributed as dist

    local_error = None
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "Internode GIN initialization cannot run during CUDA Graph capture."
            )
        layout.validate()
        _positive_int("signal_count", signal_count)
        _positive_int("ranks_per_node", ranks_per_node)
        if (
            world_size % ranks_per_node
            or world_size // ranks_per_node < 2
            or ranks_per_node > 16
        ):
            raise ValueError("GIN requires >=2 domains with R<=16 dividing world_size.")
        if not dist.is_initialized():
            raise ValueError("the default process group must already be initialized.")
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise ValueError(
                "GIN rank/world_size must match the default process group."
            )
        if not torch.cuda.is_available() or torch.cuda.current_device() != device:
            raise ValueError(
                "bind the bootstrap CUDA device before entering GIN initialization."
            )
        require_gin_apis()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    agreed_signature = (
        world_size,
        ranks_per_node,
        layout.total_bytes,
        tuple((r.name, r.offset, r.nbytes) for r in layout.regions),
        signal_count,
        signature,
    )
    collective_gin_preflight(
        dist,
        signature=agreed_signature,
        local_error=local_error,
        phase="before-communicator",
        requested_group=process_group,
    )
    from nccl.core import (
        Communicator,
        NCCLDevCommRequirements,
        NcclGinConnectionType,
        UniqueId,
        WindowFlag,
        get_unique_id,
    )
    from nccl.core.interop.torch import empty

    session = GinSession(device=device, layout=layout)

    def step(phase, action, *, collective=False):
        error = None
        result = None
        try:
            result = action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            collective_gin_preflight(
                dist,
                signature=agreed_signature,
                local_error=error,
                phase=phase,
            )
        except Exception as exc:
            if collective or not isinstance(exc, GinPreflightError):
                _failed_sessions.append(session)
                raise GinFatalError(
                    f"GIN {phase} failed; terminate all ranks and do not reuse this session."
                ) from exc
            try:
                session.close()
            except Exception as cleanup_error:
                _failed_sessions.append(session)
                raise GinFatalError(
                    "GIN cleanup failed; terminate all ranks."
                ) from cleanup_error
            raise
        return result

    # UID production is local and may fail on rank zero: agree before broadcast.
    payload = [step("unique-id", lambda: bytes(get_unique_id()) if rank == 0 else None)]
    step(
        "unique-id-broadcast",
        lambda: dist.broadcast_object_list(
            payload,
            src=0,
            group=dist.group.WORLD,
            device=torch.device("cuda", device),
        ),
        collective=True,
    )
    import hashlib

    session.identity = hashlib.sha256(payload[0]).hexdigest()
    uid = step("unique-id-decode", lambda: UniqueId.from_bytes(payload[0]))

    def create_comm():
        session.comm = Communicator.init(world_size, rank, uid)

    step("communicator", create_comm, collective=True)
    step(
        "rail-capability",
        lambda: validate_gin_rail_support(
            session.comm,
            world_size=world_size,
            ranks_per_node=ranks_per_node,
        ),
    )

    def allocate():
        session.root = empty((layout.total_bytes,), dtype=torch.uint8, device=device)
        session.root.zero_()
        # Finish stream work and surface allocation errors before registration.
        torch.cuda.synchronize(device)

    step("arena-allocation", allocate)

    def create_dev_comm():
        session.dev_comm = session.comm.create_dev_comm(
            NCCLDevCommRequirements(
                gin_signal_count=signal_count,
                gin_connection_type=NcclGinConnectionType.RAIL,
                gin_strong_signals_required=False,
                gin_va_signals_required=False,
            )
        )
        if session.dev_comm is None or not int(session.dev_comm.ptr):
            raise RuntimeError("NCCL returned a null GIN device communicator.")

    step("device-communicator", create_dev_comm, collective=True)

    def register():
        session.window = session.comm.register_window(
            session.root, WindowFlag.COLL_SYMMETRIC
        )
        if session.window is None or not int(session.window.handle):
            raise RuntimeError("NCCL returned a null registered window.")

    step("window-registration", register, collective=True)

    def map_peers():
        first = rank // ranks_per_node * ranks_per_node
        addresses = tuple(
            (peer, session.window.get_peer_device_pointer(peer))
            for peer in range(first, first + ranks_per_node)
        )
        session.peer_offsets = lsa_peer_offsets(
            rank=rank,
            world_size=world_size,
            ranks_per_node=ranks_per_node,
            base_address=int(session.root.data_ptr()),
            peer_addresses=tuple(
                (p, int(a) if a is not None else 0) for p, a in addresses
            ),
        )

    step("lsa-peer-mapping", map_peers)
    return session
