"""Host-side NCCL GIN runtime helpers for MegaMoE internode bring-up.

This module deliberately stays out of the default runner path.  It creates the
NCCL device communicator and registered windows required by CuTe GIN kernels,
but does not issue device-side GIN operations by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


class NcclGinUnavailableError(RuntimeError):
    """Raised when the communicator cannot provide requested GIN resources."""


@dataclass(frozen=True)
class NcclGinCapabilities:
    """Installed nccl4py/CuTe GIN API surface relevant to this project."""

    available: bool
    error: Optional[str]
    has_cute_put: bool
    has_cute_signal: bool
    has_cute_wait_signal: bool
    has_cute_flush: bool
    has_cute_read_signal: bool
    has_cute_reset_signal: bool
    has_cute_signal_shadow_pointer: bool
    has_rail_connection_type: bool
    has_full_connection_type: bool

    @property
    def release_wheel_minimal_ok(self) -> bool:
        return (
            self.available
            and self.has_cute_put
            and self.has_cute_signal
            and self.has_cute_wait_signal
            and self.has_cute_read_signal
            and self.has_cute_reset_signal
            and self.has_cute_signal_shadow_pointer
            and self.has_rail_connection_type
        )


@dataclass
class NcclLsaContext:
    """Own an NCCL communicator used only for LSA window registration."""

    comm: object

    def close(self) -> None:
        close = getattr(self.comm, "close", None)
        if close is not None:
            close()
            return
        destroy = getattr(self.comm, "destroy", None)
        if destroy is not None:
            destroy()


@dataclass
class NcclLsaRuntime:
    """Own an LSA registered window and its communicator."""

    context: NcclLsaContext
    window: object

    def close(self) -> None:
        try:
            close = getattr(self.window, "close", None)
            if close is not None:
                close()
                return
            destroy = getattr(self.window, "destroy", None)
            if destroy is not None:
                destroy()
        finally:
            self.context.close()


@dataclass
class NcclGinContext:
    """Owns NCCL communicator resources independent of registered windows."""

    comm: object
    dev_comm: object
    connection_type: str
    signal_count: int
    barrier_count: int
    gin_force_enable: bool
    support: "NcclGinRuntimeSupport"

    @property
    def dev_comm_ptr(self) -> int:
        return int(self.dev_comm.ptr)

    def close(self) -> None:
        """Release NCCL resources in reverse dependency order."""
        for resource in (self.dev_comm, self.comm):
            close = getattr(resource, "close", None)
            if close is not None:
                close()
                continue
            destroy = getattr(resource, "destroy", None)
            if destroy is not None:
                destroy()


@dataclass
class NcclGinRuntime:
    """Owns all NCCL GIN resources needed by a launch."""

    context: NcclGinContext
    window: object

    @property
    def comm(self) -> object:
        return self.context.comm

    @property
    def dev_comm(self) -> object:
        return self.context.dev_comm

    @property
    def dev_comm_ptr(self) -> int:
        return self.context.dev_comm_ptr

    @property
    def window_handle(self) -> int:
        return int(self.window.handle)

    @property
    def connection_type(self) -> str:
        return self.context.connection_type

    @property
    def signal_count(self) -> int:
        return self.context.signal_count

    @property
    def barrier_count(self) -> int:
        return self.context.barrier_count

    @property
    def gin_force_enable(self) -> bool:
        return self.context.gin_force_enable

    def close(self) -> None:
        """Release window and communicator resources in dependency order."""
        try:
            close = getattr(self.window, "close", None)
            if close is not None:
                close()
                return
            destroy = getattr(self.window, "destroy", None)
            if destroy is not None:
                destroy()
        finally:
            self.context.close()


@dataclass(frozen=True)
class NcclGinRuntimeSupport:
    """Runtime GIN support reported by an initialized NCCL communicator."""

    device_api_support: bool
    gin_type: str
    railed_gin_type: str
    n_lsa_teams: int
    host_rma_support: bool
    multimem_support: bool

    @property
    def full_gin_available(self) -> bool:
        return self.gin_type != "NONE"

    @property
    def rail_gin_available(self) -> bool:
        return self.railed_gin_type != "NONE"


def detect_nccl_gin_capabilities() -> NcclGinCapabilities:
    """Inspect the installed nccl4py CuTe GIN wrapper without touching CUDA."""
    try:
        from nccl.core import NcclGinConnectionType
        from nccl.core.device.cute import Gin
    except Exception as exc:  # pragma: no cover - environment dependent.
        return NcclGinCapabilities(
            available=False,
            error=f"{type(exc).__name__}: {exc}",
            has_cute_put=False,
            has_cute_signal=False,
            has_cute_wait_signal=False,
            has_cute_flush=False,
            has_cute_read_signal=False,
            has_cute_reset_signal=False,
            has_cute_signal_shadow_pointer=False,
            has_rail_connection_type=False,
            has_full_connection_type=False,
        )

    return NcclGinCapabilities(
        available=True,
        error=None,
        has_cute_put=hasattr(Gin, "put"),
        has_cute_signal=hasattr(Gin, "signal"),
        has_cute_wait_signal=hasattr(Gin, "wait_signal"),
        has_cute_flush=hasattr(Gin, "flush"),
        has_cute_read_signal=hasattr(Gin, "read_signal"),
        has_cute_reset_signal=hasattr(Gin, "reset_signal"),
        has_cute_signal_shadow_pointer=hasattr(
            Gin, "signal_shadow_pointer"
        ),
        has_rail_connection_type=hasattr(NcclGinConnectionType, "RAIL"),
        has_full_connection_type=hasattr(NcclGinConnectionType, "FULL"),
    )


def preflight_nccl_gin_runtime() -> NcclGinCapabilities:
    """Validate import-level NCCL GIN support before allocating windows.

    This is intentionally limited to Python/CuTe API availability.  A usable
    device communicator and registered window still require a CUDA allocation,
    so they are attached later once the kernel-derived window layout is known.
    """
    caps = detect_nccl_gin_capabilities()
    if not caps.release_wheel_minimal_ok:
        raise NcclGinUnavailableError(
            "Installed nccl4py is missing required CuTe GIN release-wheel APIs: "
            f"{caps}."
        )
    return caps


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return str(value)


def _query_nccl_gin_runtime_support(comm: object) -> NcclGinRuntimeSupport:
    return NcclGinRuntimeSupport(
        device_api_support=bool(comm.device_api_support),
        gin_type=_enum_name(comm.gin_type),
        railed_gin_type=_enum_name(comm.railed_gin_type),
        n_lsa_teams=int(comm.n_lsa_teams),
        host_rma_support=bool(comm.host_rma_support),
        multimem_support=bool(comm.multimem_support),
    )


def _validate_nccl_gin_runtime_support(
    support: NcclGinRuntimeSupport,
    *,
    connection_type: str,
) -> None:
    if not support.device_api_support:
        raise NcclGinUnavailableError(
            "NCCL communicator reports device_api_support=False; cannot "
            f"create a GIN dev communicator. support={support}"
        )
    if connection_type == "rail" and not support.rail_gin_available:
        raise NcclGinUnavailableError(
            "NCCL communicator reports railed GIN is unavailable; cannot "
            f"use connection_type='rail'. support={support}"
        )
    if connection_type == "full" and not support.full_gin_available:
        raise NcclGinUnavailableError(
            "NCCL communicator reports full GIN is unavailable; cannot use "
            f"connection_type='full'. support={support}"
        )


def validate_nccl_gin_rail_topology(
    support: NcclGinRuntimeSupport,
    *,
    world_size: int,
    ranks_per_node: int,
) -> None:
    """Fail closed unless NCCL's rail decomposition matches logical nodes.

    team_rail contains one same-LSA-rank peer per LSA team. MegaMoE uses
    its peer index as a logical node index, so the communicator must expose
    exactly world_size / ranks_per_node LSA teams. A same-NVL-domain
    allocation that collapses the requested logical nodes into one LSA team
    is therefore not a valid Stage 1 topology, even if GIN itself initializes.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if ranks_per_node <= 0:
        raise ValueError(
            f"ranks_per_node must be positive, got {ranks_per_node}."
        )
    if world_size % ranks_per_node != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"ranks_per_node ({ranks_per_node})."
        )
    expected_teams = world_size // ranks_per_node
    if expected_teams < 2:
        raise NcclGinUnavailableError(
            "MegaMoE Stage 1 requires at least two logical nodes; "
            f"world_size={world_size}, ranks_per_node={ranks_per_node}."
        )
    if support.n_lsa_teams != expected_teams:
        raise NcclGinUnavailableError(
            "NCCL rail topology does not match MegaMoE logical nodes: "
            f"n_lsa_teams={support.n_lsa_teams}, expected={expected_teams} "
            f"for world_size={world_size}, ranks_per_node={ranks_per_node}. "
            "Use ranks from distinct NVL domains (nvlblk values); do not "
            "silently run the rail path inside one collapsed LSA team."
        )


def _broadcast_unique_id_bytes(rank: int, world_size: int) -> bytes:
    from nccl.core import get_unique_id, UniqueId

    if world_size == 1:
        return bytes(get_unique_id())

    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        raise RuntimeError(
            "NCCL GIN runtime requires torch.distributed for multi-rank "
            "unique-id broadcast."
        )

    uid_bytes = bytes(get_unique_id()) if rank == 0 else None
    payload = [uid_bytes]
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("rank 0 did not broadcast an NCCL unique id.")
    # Validate reconstruction before returning, so bad transports fail here.
    UniqueId.from_bytes(payload[0])
    return payload[0]


def create_nccl_lsa_context(*, rank: int, world_size: int) -> NcclLsaContext:
    """Create a plain NCCL communicator for LSA without GIN resources."""
    from nccl.core import Communicator, UniqueId

    uid = UniqueId.from_bytes(_broadcast_unique_id_bytes(rank, world_size))
    return NcclLsaContext(comm=Communicator.init(world_size, rank, uid))


def register_nccl_lsa_window(
    context: NcclLsaContext,
    window_tensor: torch.Tensor,
) -> NcclLsaRuntime:
    """Collectively register one symmetric LSA arena on ``context``."""
    if not window_tensor.is_cuda:
        raise ValueError("NCCL LSA windows must register CUDA tensors.")

    from nccl.core import WindowFlag

    window = context.comm.register_window(
        window_tensor, WindowFlag.COLL_SYMMETRIC,
    )
    if window is None:
        raise RuntimeError("NCCL returned a null LSA registered-window handle.")
    return NcclLsaRuntime(context=context, window=window)


def create_nccl_gin_context(
    *,
    rank: int,
    world_size: int,
    signal_count: int,
    barrier_count: int,
    connection_type: str = "rail",
    context_count: Optional[int] = None,
    strong_signals_required: bool = False,
    gin_va_signals_required: bool = False,
    gin_force_enable: bool = False,
    skip_support_precheck: bool = False,
) -> NcclGinContext:
    """Create an NCCL communicator and GIN device communicator.

    Window registration is deliberately a separate step: NCCL GIN devComm setup
    does not require a buffer, while ``register_nccl_gin_window`` must wait
    until the caller has allocated the symmetric CUDA window tensor.
    """
    if signal_count <= 0:
        raise ValueError(f"signal_count must be positive, got {signal_count}.")
    if barrier_count < 0:
        raise ValueError(
            f"barrier_count must be non-negative, got {barrier_count}."
        )

    preflight_nccl_gin_runtime()

    from nccl.core import (
        Communicator,
        NCCLDevCommRequirements,
        NcclGinConnectionType,
        UniqueId,
    )

    connection_key = connection_type.lower()
    if connection_key not in ("rail", "full"):
        raise ValueError(
            f"connection_type must be 'rail' or 'full', got {connection_type!r}."
        )
    gin_connection_type = (
        NcclGinConnectionType.RAIL
        if connection_key == "rail"
        else NcclGinConnectionType.FULL
    )

    uid = UniqueId.from_bytes(_broadcast_unique_id_bytes(rank, world_size))
    comm = Communicator.init(world_size, rank, uid)
    support = _query_nccl_gin_runtime_support(comm)

    req_kwargs = {
        "gin_signal_count": signal_count,
        "gin_connection_type": gin_connection_type,
        "gin_strong_signals_required": strong_signals_required,
        "gin_va_signals_required": gin_va_signals_required,
    }
    if gin_force_enable:
        req_kwargs["gin_force_enable"] = True
    if context_count is not None:
        if context_count <= 0:
            raise ValueError(
                f"context_count must be positive when set, got {context_count}."
            )
        req_kwargs["gin_context_count"] = context_count
    # The production MegaMoE barrier is built from two ordinary signal slots
    # so it can self-cancel under Nsight Compute kernel replay. Do not request
    # unused library barrier objects when barrier_count == 0; standalone
    # runtime smokes can still request them explicitly.
    if barrier_count:
        if connection_key == "rail":
            req_kwargs["rail_gin_barrier_count"] = barrier_count
        else:
            req_kwargs["world_gin_barrier_count"] = barrier_count

    try:
        if not (gin_force_enable or skip_support_precheck):
            _validate_nccl_gin_runtime_support(
                support,
                connection_type=connection_key,
            )
        dev_comm = comm.create_dev_comm(NCCLDevCommRequirements(**req_kwargs))
    except Exception as exc:
        comm.destroy()
        if gin_force_enable or skip_support_precheck:
            raise NcclGinUnavailableError(
                "NCCL could not create a GIN dev communicator; "
                f"support={support} requirements={req_kwargs}"
            ) from exc
        raise

    return NcclGinContext(
        comm=comm,
        dev_comm=dev_comm,
        connection_type=connection_key,
        signal_count=signal_count,
        barrier_count=barrier_count,
        gin_force_enable=gin_force_enable,
        support=support,
    )


def register_nccl_gin_window(
    context: NcclGinContext,
    window_tensor: torch.Tensor,
) -> NcclGinRuntime:
    """Register ``window_tensor`` and combine it with an existing GIN context.

    ``window_tensor`` must have the same shape and dtype on every rank because
    NCCL window registration is collective.  The caller keeps the returned
    runtime and its context alive until all kernels using the handles finish.
    """
    if not window_tensor.is_cuda:
        raise ValueError("NCCL GIN windows must register CUDA tensors.")

    from nccl.core import WindowFlag

    window = context.comm.register_window(window_tensor, WindowFlag.COLL_SYMMETRIC)
    if window is None:
        raise RuntimeError("NCCL returned a null registered-window handle.")

    return NcclGinRuntime(
        context=context,
        window=window,
    )
