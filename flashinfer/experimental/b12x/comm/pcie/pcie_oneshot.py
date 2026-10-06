"""PCIe oneshot allreduce runtime with optional crossover autotuning."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import count
from threading import RLock
from typing import Callable, Iterable, Optional, Sequence, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation.types import Plan, require_prepared
from ._cuda_ipc import CudaRTLibrary


logger = logging.getLogger(__name__)

SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 10)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DEFAULT_MAX_SIZE = 8 * 1024 * 1024
DEFAULT_RANK_DATA_BYTES = 8 * 1024 * 1024
TP2_PLAIN_REMOTE_PUSH_MAX_ROWS = 512
TP2_PLAIN_REMOTE_PUSH_AUTO_MAX_ROWS = 64
TP2_PLAIN_REMOTE_PUSH_HIDDEN_SIZE = 4096
TP2_PLAIN_REMOTE_PUSH_MAX_BYTES = (
    TP2_PLAIN_REMOTE_PUSH_MAX_ROWS
    * TP2_PLAIN_REMOTE_PUSH_HIDDEN_SIZE
    * torch.bfloat16.itemsize
)
TP2_PLAIN_REMOTE_PUSH_AUTO_MAX_BYTES = (
    TP2_PLAIN_REMOTE_PUSH_AUTO_MAX_ROWS
    * TP2_PLAIN_REMOTE_PUSH_HIDDEN_SIZE
    * torch.bfloat16.itemsize
)
AUTOTUNE_CEILING = 1 * 1024 * 1024
AUTOTUNE_FINE_STEP = 8 * 1024
IPC_SLAB_ALIGNMENT = 256
ONESHOT_REQUIRED_SMS = 36
_SINGLE_CHANNEL_ID = "__single__"

# CUDA graph nodes and the native runtime retain raw pointers into rank_data.
# GC cannot synchronize arbitrary streams, so an abandoned runtime must keep
# its complete Python ownership graph alive until process teardown.  Explicit
# close() is the only path that removes this need.  A dict makes manual/finalizer
# re-entry idempotent while the retained object prevents id reuse.
_ABANDONED_PCIE_RUNTIME_QUARANTINE: dict[int, object] = {}
_RETAINED_FAILED_IPC_EXPORTS: dict[tuple[int, int], "_RetryableIPCExport"] = {}
_RETAINED_IPC_SETUP_GENERATIONS = count(1)
_SetupResult = TypeVar("_SetupResult")


def parse_pcie_oneshot_max_size(value: str | int | None) -> Optional[int]:
    """Parse a byte-size string, or return ``None`` for ``auto``."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if value.lower() == "auto":
        return None
    normalized = value.upper().strip()
    suffixes = {
        "KB": 1024,
        "K": 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
    }
    for suffix, multiplier in sorted(suffixes.items(), key=lambda item: -len(item[0])):
        if normalized.endswith(suffix):
            return int(normalized[: -len(suffix)]) * multiplier
    return int(value)


def _normalize_device(device: torch.device | int | str) -> torch.device:
    if isinstance(device, int):
        device_obj = torch.device(f"cuda:{device}")
    elif isinstance(device, torch.device):
        device_obj = device
    else:
        device_obj = torch.device(device)
    if device_obj.type == "cuda" and device_obj.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device_obj


def _device_guard(device: torch.device | int | str):
    """Match the native entry points' OptionalCUDAGuard semantics."""

    device_obj = _normalize_device(device)
    if device_obj.type != "cuda":
        return nullcontext()
    return torch.cuda.device(device_obj)


def _cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("expected a CUDA device")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _current_stream_key(
    device: torch.device | int | str, stream: object = None
) -> Optional[int]:
    device_obj = _normalize_device(device)
    if device_obj.type != "cuda":
        return None
    if stream is None:
        stream = torch.cuda.current_stream(device_obj)
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    return int(stream)


def _push_mode_enabled() -> bool:
    """Match the extension's B12X_PCIE_ONESHOT_PUSH transport toggle.

    Push transport writes each rank's input into a per-source shard of every
    peer's eager slot, so each slot must hold world_size * max_size bytes.
    """

    return os.getenv("B12X_PCIE_ONESHOT_PUSH", "0") not in ("", "0")


def _tp2_remote_push_enabled() -> bool:
    """Enable the qualified fused TP2 remote-write transport."""

    return os.getenv("B12X_PCIE_TP2_REMOTE_PUSH", "0") not in ("", "0")


def _tp2_plain_remote_push_enabled() -> bool:
    """Enable the qualified plain TP2 peer-push transport by default.

    ``B12X_PCIE_TP2_REMOTE_PUSH`` remains a compatible global TP2 override
    when the plain-specific variable is absent. This preserves explicit
    deployment controls without changing the fused transport's default.
    """

    configured = os.getenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH")
    if configured is None:
        configured = os.getenv("B12X_PCIE_TP2_REMOTE_PUSH", "1")
    return configured not in ("", "0")


def _tp4_remote_push_enabled() -> bool:
    """Enable the qualified fused TP4 remote-write transport."""

    return os.getenv("B12X_PCIE_TP4_REMOTE_PUSH", "0") not in ("", "0")


def _tp8_owner_reduce_enabled() -> bool:
    """Enable the qualified topology-scoped fused TP8 transport by default."""

    return os.getenv("B12X_PCIE_TP8_OWNER_REDUCE", "1") not in ("", "0")


def _uses_sharded_eager_storage(
    world_size: int,
    transport_policy: Optional[tuple[bool, bool, bool, bool, bool]] = None,
) -> bool:
    """Return whether staged fused transport needs one shard per source."""

    push, tp2_remote, _, tp4_remote, tp8_owner = (
        _transport_policy_contract() if transport_policy is None else transport_policy
    )
    return push or (
        (world_size == 2 and tp2_remote)
        or (world_size == 4 and tp4_remote)
        or (world_size == 8 and tp8_owner)
    )


def _eager_storage_shards(
    world_size: int,
    transport_policy: Optional[tuple[bool, bool, bool, bool, bool]] = None,
) -> int:
    """Return fixed-size shards required by each eager staging slot.

    Topology transports reserve one source shard per rank. TP2 plain
    peer-push also reserves one isolated incoming shard so its unpublished
    markers cannot alias payloads retained by fused or pull collectives.
    """

    policy = (
        _transport_policy_contract() if transport_policy is None else transport_policy
    )
    base_shards = world_size if _uses_sharded_eager_storage(world_size, policy) else 1
    # Graph peer-push uses a data-and-epoch layout. Every 16 bytes of input
    # occupy 32 bytes of incoming scratch so readiness is carried in the same
    # PCIe transactions as the payload instead of a separate peer atomic.
    # Eager channels use only the first incoming shard.
    plain_tp2_incoming = 2 if world_size == 2 and policy[2] else 0
    return base_shards + plain_tp2_incoming


def _transport_policy_contract() -> tuple[bool, bool, bool, bool, bool]:
    """Values that must agree across ranks before IPC storage is allocated."""

    return (
        _push_mode_enabled(),
        _tp2_remote_push_enabled(),
        _tp2_plain_remote_push_enabled(),
        _tp4_remote_push_enabled(),
        _tp8_owner_reduce_enabled(),
    )


def _tp2_plain_remote_push_rows(inp: torch.Tensor) -> Optional[int]:
    """Return the row count for the qualified TP2 peer-push tensor contract."""

    if (
        inp.dtype not in (torch.float16, torch.bfloat16)
        or inp.ndim == 0
        or int(inp.shape[-1]) != TP2_PLAIN_REMOTE_PUSH_HIDDEN_SIZE
        or inp.numel() % TP2_PLAIN_REMOTE_PUSH_HIDDEN_SIZE != 0
    ):
        return None
    rows = inp.numel() // TP2_PLAIN_REMOTE_PUSH_HIDDEN_SIZE
    if rows < 1 or rows > TP2_PLAIN_REMOTE_PUSH_MAX_ROWS:
        return None
    return rows


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    """Return whether raw-pointer traversal visits every logical element once."""

    if inp.is_contiguous():
        return True
    expected_stride = 1
    dimensions = []
    for size, stride in zip(inp.shape, inp.stride(), strict=True):
        size = int(size)
        stride = int(stride)
        if size <= 1:
            continue
        if stride < 0:
            return False
        dimensions.append((stride, size))
    for stride, size in sorted(dimensions):
        if stride != expected_stride:
            return False
        expected_stride *= size
    storage = inp.untyped_storage()
    return (
        storage.nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _resolve_exchange_group(
    exchange_group: Optional[ProcessGroup],
    process_group: Optional[ProcessGroup],
) -> Optional[ProcessGroup]:
    if (
        exchange_group is not None
        and process_group is not None
        and exchange_group is not process_group
    ):
        raise ValueError("pass only one of exchange_group or process_group")
    return exchange_group if exchange_group is not None else process_group


def _normalize_logical_channel_id(channel_id: str) -> str:
    if not isinstance(channel_id, str):
        raise TypeError("logical channel id must be a string")
    normalized = channel_id.strip()
    if not normalized:
        raise ValueError("logical channel id must not be empty")
    return normalized


def _collective_capture_needs_preparation(
    *,
    owner: str,
    logical_id: str,
    prepared_channel_ids: Iterable[str],
    exchange_group: ProcessGroup,
) -> bool:
    """Decide whether capture may use the prepared catalog without allocation.

    Once every rank has prepared the same complete graph catalog, ranks may
    capture those graphs in different orders.  Allocation remains collective:
    a same-id unknown channel may use the convenience preparation path, while
    any differing request that includes an unknown id fails before allocation.
    """

    catalog = tuple(sorted(prepared_channel_ids))
    local_state = (logical_id, catalog)
    gathered = _broadcast_gather_object(local_state, exchange_group)
    if not gathered:
        raise RuntimeError(f"{owner} capture contract returned no rank states")

    requested_ids: list[str] = []
    for group_rank, state in enumerate(gathered):
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise RuntimeError(
                f"invalid {owner} capture contract from group rank {group_rank}"
            )
        requested_id, peer_catalog = state
        if not isinstance(requested_id, str) or not isinstance(
            peer_catalog, (tuple, list)
        ):
            raise RuntimeError(
                f"invalid {owner} capture contract from group rank {group_rank}"
            )
        peer_catalog = tuple(peer_catalog)
        if any(not isinstance(value, str) for value in peer_catalog):
            raise RuntimeError(
                f"invalid {owner} capture contract from group rank {group_rank}"
            )
        if peer_catalog != catalog:
            raise RuntimeError(
                f"{owner} prepared channel catalog differs across ranks: {gathered}"
            )
        requested_ids.append(requested_id)

    if all(requested_id in catalog for requested_id in requested_ids):
        # The canonical catalog already fixed allocation order.  Current graph
        # capture order is process-local and need not agree across ranks.
        return False
    if all(requested_id == requested_ids[0] for requested_id in requested_ids):
        # Preserve the collective same-id convenience allocation path.
        return True
    raise RuntimeError(
        f"{owner} capture requested unprepared logical channels in differing "
        f"rank order: requested={tuple(requested_ids)}, prepared={catalog}; call "
        "prepare_channels() collectively with the complete graph set first"
    )


def _group_ranks(group: ProcessGroup) -> list[int]:
    world_size = dist.get_world_size(group=group)
    if hasattr(dist, "get_process_group_ranks"):
        # Process-group rank is an index into the order returned here.  Sorting
        # silently changes that mapping for non-monotonic groups and makes the
        # handle in slot N belong to a different peer than group rank N.
        ranks = list(dist.get_process_group_ranks(group=group))
        if len(ranks) != world_size:
            raise RuntimeError("process-group rank list does not match world size")
        return ranks
    if hasattr(dist, "get_global_rank"):
        return [
            dist.get_global_rank(group, group_rank) for group_rank in range(world_size)
        ]
    return list(range(world_size))


def _object_broadcast_device(group: ProcessGroup) -> torch.device | str:
    try:
        try:
            backend = dist.get_backend(group=group)
        except TypeError:
            backend = dist.get_backend(group)
    except Exception as exc:
        raise RuntimeError(
            "PCIe object exchange requires a valid process group"
        ) from exc
    backend_name = str(backend).lower()
    if "gloo" in backend_name:
        return torch.device("cpu")
    if "nccl" not in backend_name:
        raise RuntimeError(
            "PCIe object exchange requires an NCCL or Gloo process group, "
            f"got {backend}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PCIe oneshot IPC exchange requires CUDA")
    return torch.device("cuda", torch.cuda.current_device())


def _broadcast_gather_object(local_object: object, group: ProcessGroup) -> list[object]:
    # `broadcast_object_list` is more robust than `all_gather_object` for
    # the host-side exchange that happens around IPC setup and graph capture.
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    all_objects: list[list[object | None]] = [[None] for _ in range(world_size)]
    all_objects[rank][0] = local_object
    device = _object_broadcast_device(group)
    for index, src_rank in enumerate(_group_ranks(group)):
        dist.broadcast_object_list(
            all_objects[index], src=src_rank, group=group, device=device
        )
    return [entry[0] for entry in all_objects]


@dataclass(frozen=True)
class _OwnedSharedBuffer:
    local_ptr: int
    peer_ptrs: tuple[int, ...]
    remote_ptrs: tuple[int, ...]


@dataclass
class _RetryableIPCExport:
    """Own a failed collective IPC setup until a coordinated retry succeeds.

    The historical name is retained for compatibility with callers that only
    need to retry a failed ``cudaFree``.  A setup can also own still-open peer
    imports, a native runtime cleanup callback, and a collective participation
    ticket after this rank has already released its local export.  Keeping the
    ticket is important: a rank that succeeded locally must still join the
    retry verdict so a peer can safely finish its rollback.
    """

    ipc: CudaRTLibrary
    local_ptr: int
    exchange_group: Optional[ProcessGroup] = None
    owner: str = "PCIe shared buffer"
    phase: str = "rollback"
    remote_ptrs: list[int] = field(default_factory=list)
    local_cleanup: Optional[Callable[[], None]] = None
    state: str = "free"
    _registry_key: tuple[int, int] | None = None

    @property
    def key(self) -> tuple[int, int]:
        if self._registry_key is not None:
            return self._registry_key
        return id(self.ipc), int(self.local_ptr)

    def retry(self) -> None:
        key = self.key
        if key not in _RETAINED_FAILED_IPC_EXPORTS:
            return

        if self.state == "native":
            native_error: BaseException | None = None
            if self.local_cleanup is not None:
                try:
                    self.local_cleanup()
                except Exception as exc:
                    native_error = exc
                else:
                    self.local_cleanup = None

            if self.exchange_group is None:
                if native_error is not None:
                    raise RuntimeError(
                        f"{self.owner} {self.phase} retry native cleanup failed; "
                        "CUDA IPC ownership remains retained"
                    ) from native_error
            else:
                native_statuses = _exchange_setup_failures(
                    native_error,
                    exchange_group=self.exchange_group,
                    phase=f"{self.phase} retry native cleanup",
                )
                if any(native_statuses):
                    raise RuntimeError(
                        _setup_failure_message(
                            self.owner,
                            f"{self.phase} retry native cleanup",
                            native_statuses,
                            exports_retained=True,
                        )
                    ) from native_error
            self.state = "unmap"

        if self.state == "unmap":
            failures: list[str] = []

            remaining_remote_ptrs: list[int] = []
            for ptr in self.remote_ptrs:
                try:
                    self.ipc.cudaIpcCloseMemHandle(ptr)
                except Exception as exc:
                    remaining_remote_ptrs.append(ptr)
                    failures.append(
                        f"CUDA IPC import {ptr}: {type(exc).__name__}: {exc}"
                    )
            self.remote_ptrs = remaining_remote_ptrs

            local_error = RuntimeError(" | ".join(failures)) if failures else None
            if self.exchange_group is None:
                if local_error is not None:
                    raise RuntimeError(
                        f"{self.owner} {self.phase} retry failed; CUDA IPC "
                        "ownership remains retained"
                    ) from local_error
            else:
                statuses = _exchange_setup_failures(
                    local_error,
                    exchange_group=self.exchange_group,
                    phase=f"{self.phase} retry unmap",
                )
                if any(statuses):
                    raise RuntimeError(
                        _setup_failure_message(
                            self.owner,
                            f"{self.phase} retry unmap",
                            statuses,
                            exports_retained=True,
                        )
                    ) from local_error
            self.state = "free"

        free_error: BaseException | None = None
        if self.local_ptr:
            ptr = self.local_ptr
            try:
                self.ipc.cudaFree(ptr)
            except Exception as exc:
                free_error = exc
            else:
                self.local_ptr = 0

        if self.exchange_group is None:
            if free_error is not None:
                raise RuntimeError(
                    f"{self.owner} {self.phase} retry export free failed; CUDA "
                    "IPC ownership remains retained"
                ) from free_error
        else:
            free_statuses = _exchange_setup_failures(
                free_error,
                exchange_group=self.exchange_group,
                phase=f"{self.phase} retry export free",
            )
            if any(free_statuses):
                raise RuntimeError(
                    _setup_failure_message(
                        self.owner,
                        f"{self.phase} retry export free",
                        free_statuses,
                        exports_retained=True,
                    )
                ) from free_error

        _RETAINED_FAILED_IPC_EXPORTS.pop(key, None)


def _retain_failed_ipc_export(
    ipc: CudaRTLibrary, local_ptr: int
) -> _RetryableIPCExport:
    key = id(ipc), int(local_ptr)
    retained = _RETAINED_FAILED_IPC_EXPORTS.get(key)
    if retained is None:
        retained = _RetryableIPCExport(
            ipc=ipc,
            local_ptr=int(local_ptr),
            _registry_key=key,
        )
        _RETAINED_FAILED_IPC_EXPORTS[key] = retained
    return retained


def _retain_failed_ipc_setup(
    *,
    ipc: CudaRTLibrary,
    local_ptr: int,
    exchange_group: ProcessGroup,
    owner: str,
    phase: str,
    remote_ptrs: Sequence[int] = (),
    local_cleanup: Optional[Callable[[], None]] = None,
    state: str = "unmap",
) -> _RetryableIPCExport:
    """Retain every local resource and the rank's collective retry ticket."""

    # A CUDA address is not an attempt identity: it may already have been freed
    # on this rank while a peer still owns its export, then be reused by CUDA for
    # a second failed setup.  Every collective failure therefore receives an
    # independent process-local generation ticket.  The ticket need not match
    # across ranks; it only keeps this rank participating in the same retry call.
    key = id(ipc), -next(_RETAINED_IPC_SETUP_GENERATIONS)
    retained = _RetryableIPCExport(
        ipc=ipc,
        local_ptr=int(local_ptr),
        exchange_group=exchange_group,
        owner=owner,
        phase=phase,
        remote_ptrs=list(dict.fromkeys(int(ptr) for ptr in remote_ptrs)),
        local_cleanup=local_cleanup,
        state=state,
        _registry_key=key,
    )
    _RETAINED_FAILED_IPC_EXPORTS[key] = retained
    return retained


def _attach_retryable_setup(
    error: RuntimeError, retained: _RetryableIPCExport
) -> RuntimeError:
    # Keep the old attribute for downstream code/tests and expose the broader
    # meaning under a name that does not imply only cudaFree ownership.
    error.retryable_export = retained  # type: ignore[attr-defined]
    error.retryable_setup = retained  # type: ignore[attr-defined]
    return error


def _require_no_retained_ipc_setup(exchange_group: ProcessGroup) -> None:
    """Serialize collective setup generations until rollback completes.

    Retry tickets are process-local ownership objects, so allowing another IPC
    setup on the same group would make it possible for ranks to retry different
    generations in a different order.  Reject the new attempt collectively
    before it allocates anything instead.
    """

    outstanding = tuple(
        retained.key
        for retained in _RETAINED_FAILED_IPC_EXPORTS.values()
        if retained.exchange_group is exchange_group
    )
    local_error: BaseException | None = None
    if outstanding:
        local_error = RuntimeError(
            "complete the retained CUDA IPC setup retry before starting "
            f"another setup on this process group (tickets={outstanding})"
        )
    statuses = _exchange_setup_failures(
        local_error,
        exchange_group=exchange_group,
        phase="retained CUDA IPC setup gate",
    )
    if any(statuses):
        raise RuntimeError(
            _setup_failure_message(
                "PCIe shared buffer",
                "retained CUDA IPC setup gate",
                statuses,
                exports_retained=True,
            )
        ) from local_error


def _format_setup_error(error: BaseException | None) -> tuple[str, ...]:
    if error is None:
        return ()
    return (f"{type(error).__name__}: {error}",)


def _exchange_setup_failures(
    local_error: BaseException | None,
    *,
    exchange_group: ProcessGroup,
    phase: str,
    exports_retained_on_exchange_failure: bool = True,
) -> tuple[tuple[str, ...], ...]:
    """Collect a setup verdict without allowing one rank to return early."""

    try:
        gathered = _broadcast_gather_object(
            _format_setup_error(local_error), exchange_group
        )
    except Exception as exc:
        retention = (
            "; CUDA IPC exports were retained"
            if exports_retained_on_exchange_failure
            else ""
        )
        raise RuntimeError(
            f"failed to exchange peer {phase} status{retention}"
        ) from exc

    statuses: list[tuple[str, ...]] = []
    for group_rank, status in enumerate(gathered):
        if not isinstance(status, (tuple, list)):
            retention = (
                "; CUDA IPC exports were retained"
                if exports_retained_on_exchange_failure
                else ""
            )
            raise RuntimeError(
                f"invalid peer {phase} status from group rank {group_rank}{retention}"
            )
        statuses.append(tuple(str(item) for item in status))
    return tuple(statuses)


def _require_full_grid_residency(
    *,
    owner: str,
    required_sms: int,
    device: torch.device,
    exchange_group: Optional[ProcessGroup],
) -> None:
    """Reject devices where a peer-waiting worker grid may not all reside.

    The PCIe worker kernels use ``__launch_bounds__(512, 1)`` and zero dynamic
    shared memory, so every visible SM can host at least one worker CTA.  A
    device with at least the extension's maximum block count can therefore
    make the complete grid resident regardless of block scheduling order.  We
    intentionally reject smaller/MIG-like slices instead of assuming CUDA
    schedules matching block indices in the same order on every rank.
    """

    local_error: BaseException | None = None
    visible_sms = 0
    try:
        visible_sms = int(
            torch.cuda.get_device_properties(device).multi_processor_count
        )
        test_visible_sms = int(os.getenv("B12X_PCIE_TEST_VISIBLE_SM_COUNT", "0") or "0")
        if test_visible_sms > 0:
            visible_sms = min(visible_sms, test_visible_sms)
        if visible_sms < required_sms:
            raise RuntimeError(
                f"{owner} requires at least {required_sms} visible SMs so every "
                "peer-waiting CTA can be resident; this device/MIG slice exposes "
                f"{visible_sms}"
            )
    except Exception as exc:
        local_error = exc

    if exchange_group is None:
        if local_error is not None:
            raise local_error
        return

    statuses = _exchange_setup_failures(
        local_error,
        exchange_group=exchange_group,
        phase=f"{owner} resident-grid capability",
        exports_retained_on_exchange_failure=False,
    )
    if any(statuses):
        raise RuntimeError(
            _setup_failure_message(
                owner,
                "resident-grid capability",
                statuses,
                exports_retained=False,
            )
        ) from local_error


def _run_collective_preallocation_setup(
    *,
    owner: str,
    exchange_group: ProcessGroup,
    setup: Callable[[], _SetupResult],
) -> _SetupResult:
    """Run fallible local setup before any rank creates a CUDA IPC export."""

    result: Optional[_SetupResult] = None
    local_error: BaseException | None = None
    try:
        result = setup()
    except Exception as exc:
        local_error = exc

    statuses = _exchange_setup_failures(
        local_error,
        exchange_group=exchange_group,
        phase=f"{owner} pre-allocation setup",
        exports_retained_on_exchange_failure=False,
    )
    if any(statuses):
        raise RuntimeError(
            _setup_failure_message(
                owner,
                "pre-allocation setup",
                statuses,
                exports_retained=False,
            )
        ) from local_error
    assert result is not None
    return result


def _require_collective_contract(
    *, owner: str, exchange_group: ProcessGroup, contract: object
) -> None:
    """Reject divergent rank-local layout/channel inputs before allocation."""

    gathered = _broadcast_gather_object(contract, exchange_group)
    if any(peer != contract for peer in gathered):
        raise RuntimeError(f"{owner} contract differs across ranks: {gathered}")


def _finish_collective_unowned_runtime_setup(
    *,
    owner: str,
    exchange_group: ProcessGroup,
    local_error: BaseException | None,
    local_cleanup: Callable[[], None],
) -> None:
    """Coordinate direct-constructor init when pointer ownership is external."""

    statuses = _exchange_setup_failures(
        local_error,
        exchange_group=exchange_group,
        phase=f"{owner} native initialization",
        exports_retained_on_exchange_failure=False,
    )
    if not any(statuses):
        return

    cleanup_error: BaseException | None = None
    try:
        local_cleanup()
    except Exception as exc:
        cleanup_error = exc
    cleanup_statuses = _exchange_setup_failures(
        cleanup_error,
        exchange_group=exchange_group,
        phase=f"{owner} native initialization rollback",
        exports_retained_on_exchange_failure=False,
    )
    if any(cleanup_statuses):
        raise RuntimeError(
            _setup_failure_message(
                owner,
                "native initialization rollback",
                cleanup_statuses,
                exports_retained=False,
            )
        ) from (cleanup_error or local_error)
    raise RuntimeError(
        _setup_failure_message(
            owner,
            "native initialization",
            statuses,
            exports_retained=False,
        )
    ) from local_error


def _setup_failure_message(
    owner: str,
    phase: str,
    statuses: Sequence[Sequence[str]],
    *,
    exports_retained: bool,
) -> str:
    details = [
        f"group rank {group_rank}: " + " | ".join(failures)
        for group_rank, failures in enumerate(statuses)
        if failures
    ]
    retention = "; CUDA IPC exports were retained" if exports_retained else ""
    return f"{owner} {phase} failed{retention}: " + "; ".join(details)


def _abort_collective_ipc_setup(
    *,
    owner: str,
    setup_phase: str,
    setup_statuses: Sequence[Sequence[str]],
    exchange_group: ProcessGroup,
    ipc: CudaRTLibrary,
    local_ptr: int | None,
    remote_ptrs: Sequence[int],
    local_error: BaseException | None,
    local_cleanup: Optional[Callable[[], None]] = None,
) -> None:
    """Undo a failed collective IPC setup without freeing a mapped export.

    Every rank calls this function.  Each successfully opened import is closed
    exactly once.  Export ownership is released only after every group rank
    reports that all of its imports were unmapped; otherwise the raw CUDA
    allocation is deliberately retained until context teardown.
    """

    retained_cleanup = local_cleanup
    if local_cleanup is not None:
        native_error: BaseException | None = None
        try:
            local_cleanup()
        except Exception as exc:
            native_error = exc
        else:
            retained_cleanup = None
        try:
            native_statuses = _exchange_setup_failures(
                native_error,
                exchange_group=exchange_group,
                phase=f"{setup_phase} rollback native cleanup",
            )
        except Exception as exchange_error:
            assert local_ptr is not None
            retained = _retain_failed_ipc_setup(
                ipc=ipc,
                local_ptr=local_ptr,
                exchange_group=exchange_group,
                owner=owner,
                phase=setup_phase,
                remote_ptrs=remote_ptrs,
                local_cleanup=retained_cleanup,
                state="native",
            )
            error = RuntimeError(
                _setup_failure_message(
                    owner,
                    f"{setup_phase} rollback native cleanup status exchange",
                    setup_statuses,
                    exports_retained=True,
                )
            )
            raise _attach_retryable_setup(error, retained) from (
                native_error or local_error or exchange_error
            )
        if any(native_statuses):
            assert local_ptr is not None
            retained = _retain_failed_ipc_setup(
                ipc=ipc,
                local_ptr=local_ptr,
                exchange_group=exchange_group,
                owner=owner,
                phase=setup_phase,
                remote_ptrs=remote_ptrs,
                local_cleanup=retained_cleanup,
                state="native",
            )
            error = RuntimeError(
                _setup_failure_message(
                    owner,
                    f"{setup_phase} rollback native cleanup",
                    native_statuses,
                    exports_retained=True,
                )
            )
            raise _attach_retryable_setup(error, retained) from (
                native_error or local_error
            )

    unmap_failures: list[str] = []
    remaining_remote_ptrs: list[int] = []
    for ptr in remote_ptrs:
        try:
            ipc.cudaIpcCloseMemHandle(ptr)
        except Exception as exc:
            remaining_remote_ptrs.append(ptr)
            unmap_failures.append(f"CUDA IPC import {ptr}: {type(exc).__name__}: {exc}")

    try:
        unmap_statuses = _exchange_setup_failures(
            RuntimeError(" | ".join(unmap_failures)) if unmap_failures else None,
            exchange_group=exchange_group,
            phase=f"{setup_phase} rollback unmap",
        )
    except Exception as exchange_error:
        assert local_ptr is not None
        retained = _retain_failed_ipc_setup(
            ipc=ipc,
            local_ptr=local_ptr,
            exchange_group=exchange_group,
            owner=owner,
            phase=setup_phase,
            remote_ptrs=remaining_remote_ptrs,
            local_cleanup=retained_cleanup,
            state="unmap",
        )
        error = RuntimeError(
            _setup_failure_message(
                owner,
                setup_phase,
                setup_statuses,
                exports_retained=True,
            )
        )
        raise _attach_retryable_setup(error, retained) from (
            local_error or exchange_error
        )

    if any(unmap_statuses):
        assert local_ptr is not None
        retained = _retain_failed_ipc_setup(
            ipc=ipc,
            local_ptr=local_ptr,
            exchange_group=exchange_group,
            owner=owner,
            phase=setup_phase,
            remote_ptrs=remaining_remote_ptrs,
            local_cleanup=retained_cleanup,
            state="unmap",
        )
        error = RuntimeError(
            _setup_failure_message(
                owner,
                f"{setup_phase} rollback unmap",
                unmap_statuses,
                exports_retained=True,
            )
        )
        raise _attach_retryable_setup(error, retained) from local_error

    free_error: BaseException | None = None
    live_local_ptr = int(local_ptr or 0)
    if local_ptr is not None:
        try:
            ipc.cudaFree(local_ptr)
        except Exception as exc:
            free_error = exc
        else:
            live_local_ptr = 0
    try:
        free_statuses = _exchange_setup_failures(
            free_error,
            exchange_group=exchange_group,
            phase=f"{setup_phase} rollback export free",
        )
    except Exception as exchange_error:
        retained_export = _retain_failed_ipc_setup(
            ipc=ipc,
            local_ptr=live_local_ptr,
            exchange_group=exchange_group,
            owner=owner,
            phase=setup_phase,
            state="free",
        )
        error = RuntimeError(
            _setup_failure_message(
                owner,
                f"{setup_phase} rollback export free status exchange",
                setup_statuses,
                exports_retained=True,
            )
        )
        raise _attach_retryable_setup(error, retained_export) from (
            free_error or local_error or exchange_error
        )
    if any(free_statuses):
        retained_export = _retain_failed_ipc_setup(
            ipc=ipc,
            local_ptr=live_local_ptr,
            exchange_group=exchange_group,
            owner=owner,
            phase=setup_phase,
            state="free",
        )
        error = RuntimeError(
            _setup_failure_message(
                owner,
                f"{setup_phase} rollback export free",
                free_statuses,
                exports_retained=True,
            )
        )
        raise _attach_retryable_setup(error, retained_export) from (
            local_error or free_error
        )

    error = RuntimeError(
        _setup_failure_message(
            owner,
            setup_phase,
            setup_statuses,
            exports_retained=False,
        )
    )
    raise error from local_error


def _finish_collective_runtime_setup(
    *,
    owner: str,
    exchange_group: ProcessGroup,
    ipc: CudaRTLibrary,
    shared: _OwnedSharedBuffer,
    local_error: BaseException | None,
    local_cleanup: Optional[Callable[[], None]] = None,
    detach_shared_ownership: Optional[Callable[[], None]] = None,
) -> None:
    """Require every rank to construct its native runtime before returning."""

    try:
        statuses = _exchange_setup_failures(
            local_error,
            exchange_group=exchange_group,
            phase=f"{owner} native initialization",
        )
    except Exception as exchange_error:
        # A failed first verdict exchange cannot establish whether every rank
        # observed the same outcome.  Do not locally dispose, unmap, or free:
        # retain the complete native/import/export ownership set and the
        # same-group generation gate until the caller deliberately coordinates
        # retry() on every participating rank.  This is especially important
        # for twoshot, whose runtime object is not constructed until after this
        # verdict succeeds.
        retained = _retain_failed_ipc_setup(
            ipc=ipc,
            local_ptr=shared.local_ptr,
            exchange_group=exchange_group,
            owner=owner,
            phase="native initialization status exchange",
            remote_ptrs=shared.remote_ptrs,
            local_cleanup=local_cleanup,
            state="native",
        )
        if detach_shared_ownership is not None:
            detach_shared_ownership()
        error = RuntimeError(
            f"{owner} native initialization status exchange failed; CUDA IPC "
            "native/import/export ownership was retained and retry must be "
            "coordinated by every rank"
        )
        raise _attach_retryable_setup(error, retained) from exchange_error
    if any(statuses):
        try:
            _abort_collective_ipc_setup(
                owner=owner,
                setup_phase="native initialization",
                setup_statuses=statuses,
                exchange_group=exchange_group,
                ipc=ipc,
                local_ptr=shared.local_ptr,
                remote_ptrs=shared.remote_ptrs,
                local_error=local_error,
                local_cleanup=local_cleanup,
            )
        finally:
            # Rollback either released the shared allocation or transferred it
            # to a _RetryableIPCExport.  Prevent a partially constructed runtime
            # finalizer from retaining or releasing the same resources again.
            if detach_shared_ownership is not None:
                detach_shared_ownership()


def _coordinated_close_channels(
    channels: Sequence[object],
    *,
    exchange_group: Optional[ProcessGroup],
    device: torch.device,
) -> None:
    """Strictly release exports only after every peer reports successful unmap."""
    unique_channels = tuple(dict.fromkeys(channels))
    if not unique_channels:
        return

    def barrier() -> None:
        if exchange_group is None:
            return
        if device.type == "cuda":
            device_index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            dist.barrier(group=exchange_group, device_ids=[device_index])
        else:
            dist.barrier(group=exchange_group)

    if exchange_group is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        barrier()

    # Channels with strict cleanup report local unmap status before any rank
    # releases exports. Legacy channels retain their original ordered close.
    uses_strict_protocol = any(
        hasattr(channel, "_close_ipc_imports_strict") for channel in unique_channels
    )
    if not uses_strict_protocol:
        for channel in unique_channels:
            channel._close_ipc_imports()
        barrier()
        for channel in unique_channels:
            channel._free_ipc_exports()
        barrier()
        return

    unmap_errors = _run_close_phase(
        unique_channels,
        strict_method="_close_ipc_imports_strict",
        legacy_method="_close_ipc_imports",
    )
    unmap_status = _exchange_close_failures(
        tuple(_format_close_error(index, error) for index, error in unmap_errors),
        exchange_group=exchange_group,
        phase="IPC unmap",
    )
    if any(unmap_status):
        _raise_coordinated_close_error(
            "IPC unmap",
            unmap_status,
            local_errors=unmap_errors,
            exports_retained=True,
        )

    barrier()

    free_errors = _run_close_phase(
        unique_channels,
        strict_method="_free_ipc_exports_strict",
        legacy_method="_free_ipc_exports",
    )
    free_status = _exchange_close_failures(
        tuple(_format_close_error(index, error) for index, error in free_errors),
        exchange_group=exchange_group,
        phase="IPC export free",
    )
    if any(free_status):
        _raise_coordinated_close_error(
            "IPC export free",
            free_status,
            local_errors=free_errors,
            exports_retained=False,
        )

    barrier()
    for channel in unique_channels:
        if hasattr(channel, "_close_ipc_imports_strict"):
            channel._coordinated_close_complete = True


def _run_close_phase(
    channels: Sequence[object],
    *,
    strict_method: str,
    legacy_method: str,
) -> list[tuple[int, Exception]]:
    errors: list[tuple[int, Exception]] = []
    for index, channel in enumerate(channels):
        method = getattr(channel, strict_method, None)
        if method is None:
            method = getattr(channel, legacy_method)
        try:
            method()
        except Exception as exc:
            errors.append((index, exc))
    return errors


def _format_close_error(index: int, error: Exception) -> str:
    return f"channel {index}: {type(error).__name__}: {error}"


def _exchange_close_failures(
    local_failures: tuple[str, ...],
    *,
    exchange_group: Optional[ProcessGroup],
    phase: str,
) -> tuple[tuple[str, ...], ...]:
    if exchange_group is None:
        return (local_failures,)
    try:
        gathered = _broadcast_gather_object(local_failures, exchange_group)
    except Exception as exc:
        raise RuntimeError(
            f"failed to exchange peer {phase} status; exported IPC allocations "
            "were not freed"
        ) from exc

    statuses: list[tuple[str, ...]] = []
    for rank_index, status in enumerate(gathered):
        if not isinstance(status, (tuple, list)):
            raise RuntimeError(
                f"invalid peer {phase} status from group rank {rank_index}; "
                "exported IPC allocations were not freed"
            )
        statuses.append(tuple(str(item) for item in status))
    return tuple(statuses)


def _raise_coordinated_close_error(
    phase: str,
    statuses: Sequence[Sequence[str]],
    *,
    local_errors: Sequence[tuple[int, Exception]],
    exports_retained: bool,
) -> None:
    peer_details = []
    for rank_index, failures in enumerate(statuses):
        if failures:
            peer_details.append(f"group rank {rank_index}: " + " | ".join(failures))
    retention = "; exported IPC allocations were not freed" if exports_retained else ""
    error = RuntimeError(
        f"coordinated PCIe close failed during {phase}{retention}: "
        + "; ".join(peer_details)
    )
    if local_errors:
        raise error from local_errors[0][1]
    raise error


def _raise_local_cleanup_errors(
    owner: str,
    phase: str,
    failures: Sequence[tuple[str, Exception]],
) -> None:
    details = "; ".join(
        f"{resource}: {type(error).__name__}: {error}" for resource, error in failures
    )
    error = RuntimeError(f"{owner} {phase} failed: {details}")
    raise error from failures[0][1]


@dataclass(frozen=True)
class _ChannelSharedBuffers:
    owned_buffer: _OwnedSharedBuffer
    signal_ptrs: tuple[int, ...]
    eager0_ptrs: tuple[int, ...]
    eager1_ptrs: tuple[int, ...]


@dataclass(frozen=True)
class _BenchmarkResult:
    size_bytes: int
    custom_us: float
    nccl_us: float
    winner: str


@dataclass(frozen=True)
class _PlainGraphLaunchPlan:
    """Immutable dispatch contract for one prepared plain graph shape."""

    transport: str
    threads: int
    blocks: int
    size_packs: int
    stage_input: bool
    device_index: int
    eager_buffer_bytes: Optional[int]
    transport_policy: tuple[bool, bool, bool, bool, bool]
    remote_push_region_packs: int
    remote_push_slot_ptrs: tuple[int, int, int, int]


@lru_cache(maxsize=1)
def _load_extension():
    """Return the pure-Python CuTe backend used by the compatibility runtime."""

    return _CuTeOneshotBackend()


_SIGNAL_BYTES = 150_528
_POINTER_TABLE_BYTES = 16 * 8
_MAX_BLOCKS = 36
_MAX_RANKS = 16
_REG_PACKS = 3


def _pointer_as_i64(value: int) -> int:
    value = int(value)
    return value if value < (1 << 63) else value - (1 << 64)


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported oneshot dtype {dtype}") from exc


@dataclass
class _CuTeOneshotState:
    rank: int
    world_size: int
    signal_ptrs: tuple[int, ...]
    rank_data: torch.Tensor
    signal_table_address: int
    next_table_offset: int
    registered_tables: dict[int, int]
    eager_tables: Optional[tuple[int, int]] = None
    eager_ptrs: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None
    eager_buffer_bytes: Optional[int] = None
    transport_policy: tuple[bool, bool, bool, bool, bool] = (
        False,
        False,
        False,
        False,
        False,
    )
    sharded_eager_storage: bool = False
    plain_remote_push_region_packs: int = 0
    eager_slot: int = 0
    device_slot_selection: bool = False
    slot_bias: int = 0
    plain_graph_plans: dict[tuple[object, ...], _PlainGraphLaunchPlan] = field(
        default_factory=dict
    )


def _enable_device_slot_selection(
    state: _CuTeOneshotState,
    *,
    capturing: bool,
) -> None:
    """Freeze the host-to-device double-buffer phase at first capture."""
    if capturing and state.eager_tables is not None and not state.device_slot_selection:
        state.slot_bias = state.eager_slot & 1
        state.device_slot_selection = True


class _CuTeOneshotBackend:
    """Extension-shaped adapter implemented wholly in Python and CuTe DSL.

    The adapter intentionally mirrors the old pybind entry points so the
    public runtime, pooling, autotune, and test injection APIs do not need a
    second control path.  Integer handles index Python state only; no native
    object is allocated.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._next_handle = 1
        self._states: dict[int, _CuTeOneshotState] = {}

    supports_eager_storage_contract = True

    @staticmethod
    def meta_size() -> int:
        return _SIGNAL_BYTES

    def _state(self, handle: int) -> _CuTeOneshotState:
        with self._lock:
            try:
                return self._states[int(handle)]
            except KeyError as exc:
                raise RuntimeError("oneshot runtime is closed") from exc

    @staticmethod
    def _write_table(
        state: _CuTeOneshotState,
        pointers: Sequence[int],
        *,
        offset: Optional[int] = None,
    ) -> int:
        if len(pointers) != state.world_size:
            raise ValueError("pointer table must match world size")
        if offset is None:
            offset = _align_up(state.next_table_offset, _POINTER_TABLE_BYTES)
            state.next_table_offset = offset + _POINTER_TABLE_BYTES
        end = offset + _POINTER_TABLE_BYTES
        if end > state.rank_data.numel():
            raise RuntimeError("rank data buffer overflow")
        values = [_pointer_as_i64(pointer) for pointer in pointers]
        values.extend([0] * (_MAX_RANKS - len(values)))
        destination = state.rank_data[offset:end].view(torch.int64)
        source = torch.tensor(values, dtype=torch.int64, device=state.rank_data.device)
        destination.copy_(source)
        return int(state.rank_data.data_ptr()) + offset

    def init_custom_ar(
        self,
        signal_ptrs: Sequence[int],
        rank_data: torch.Tensor,
        rank: int,
    ) -> int:
        world_size = len(signal_ptrs)
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if rank_data.device.type != "cuda" or rank_data.dtype != torch.uint8:
            raise ValueError("rank_data must be a CUDA uint8 tensor")
        if rank_data.numel() < 2 * _POINTER_TABLE_BYTES:
            raise ValueError("rank_data is too small for CuTe pointer tables")

        state = _CuTeOneshotState(
            rank=int(rank),
            world_size=world_size,
            signal_ptrs=tuple(int(pointer) for pointer in signal_ptrs),
            rank_data=rank_data,
            signal_table_address=int(rank_data.data_ptr()),
            next_table_offset=_POINTER_TABLE_BYTES,
            registered_tables={},
        )
        self._write_table(state, state.signal_ptrs, offset=0)
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self._states[handle] = state
        return handle

    def register_buffer(self, handle: int, pointers: Sequence[int]) -> None:
        state = self._state(handle)
        ptrs = tuple(int(pointer) for pointer in pointers)
        local_pointer = ptrs[state.rank]
        existing = state.registered_tables.get(local_pointer)
        if existing is not None:
            return
        state.registered_tables[local_pointer] = self._write_table(state, ptrs)

    def register_pcie_buffers(
        self,
        handle: int,
        pointers0: Sequence[int],
        pointers1: Sequence[int],
        eager_buffer_bytes: Optional[int] = None,
        transport_policy: Optional[tuple[bool, bool, bool, bool, bool]] = None,
    ) -> None:
        state = self._state(handle)
        ptrs0 = tuple(int(pointer) for pointer in pointers0)
        ptrs1 = tuple(int(pointer) for pointer in pointers1)
        table0 = self._write_table(state, ptrs0)
        table1 = self._write_table(state, ptrs1)
        state.eager_tables = (table0, table1)
        state.eager_ptrs = (ptrs0, ptrs1)
        state.eager_buffer_bytes = (
            None if eager_buffer_bytes is None else int(eager_buffer_bytes)
        )
        state.transport_policy = (
            _transport_policy_contract()
            if transport_policy is None
            else tuple(bool(value) for value in transport_policy)
        )
        state.sharded_eager_storage = _uses_sharded_eager_storage(
            state.world_size,
            state.transport_policy,
        )
        if state.world_size == 2 and state.transport_policy[2]:
            base_shards = state.world_size if state.sharded_eager_storage else 1
            state.plain_remote_push_region_packs = (
                base_shards * int(state.eager_buffer_bytes or 0) // 16
            )
        else:
            state.plain_remote_push_region_packs = 0
        state.registered_tables[ptrs0[state.rank]] = table0
        state.registered_tables[ptrs1[state.rank]] = table1
        state.eager_slot = 0
        state.plain_graph_plans.clear()

    @staticmethod
    def _launch_geometry(size_packs: int) -> tuple[int, int]:
        threads = int(os.getenv("B12X_PCIE_ONESHOT_THREADS", "256"))
        threads = min(512, max(64, (threads // 32) * 32))
        block_limit = int(os.getenv("B12X_PCIE_ONESHOT_BLOCK_LIMIT", "8"))
        if block_limit <= 0 or block_limit > _MAX_BLOCKS:
            raise ValueError(
                f"B12X_PCIE_ONESHOT_BLOCK_LIMIT must be in [1, {_MAX_BLOCKS}]"
            )
        blocks = max(1, min(block_limit, (size_packs + threads - 1) // threads))
        return threads, blocks

    @staticmethod
    def _tp2_plain_remote_push_geometry(
        state: _CuTeOneshotState,
        inp: torch.Tensor,
        *,
        graph_channel: bool,
    ) -> Optional[tuple[int, int, bool]]:
        """Return TP2 peer-push geometry for a graph-owned plain all-reduce.

        The protocol requires device-selected alternating slots, which are
        owned by a prepared CUDA graph channel. Eager calls use the existing
        pull transport. The 512-row bound is functional capacity; callers use
        ``TP2_PLAIN_REMOTE_PUSH_AUTO_MAX_ROWS`` as the automatic latency policy.
        """

        if (
            state.eager_tables is None
            or state.eager_buffer_bytes is None
            or state.world_size != 2
            or not state.transport_policy[2]
            or not graph_channel
        ):
            return None
        rows = _tp2_plain_remote_push_rows(inp)
        if rows is None or inp.numel() * inp.element_size() > state.eager_buffer_bytes:
            return None
        if rows == 1:
            return 64, 36, False
        if rows == 2:
            return 128, 36, True
        if rows <= 4:
            return 256, 36, True
        if rows <= 6:
            return 64, 16, False
        if rows <= 8:
            return 256, 32, False
        if rows <= 14:
            return 64, 16, False
        if rows <= 16:
            return 64, 36, True
        if rows <= 22:
            return 128, 16, False
        if rows <= 24:
            return 128, 32, True
        return 128, 16, False

    @classmethod
    def _plain_launch_config(
        cls,
        state: _CuTeOneshotState,
        inp: torch.Tensor,
        *,
        graph_channel: bool = False,
    ) -> tuple[str, int, int]:
        """Select a plain all-reduce transport and its launch geometry."""

        remote_push = cls._tp2_plain_remote_push_geometry(
            state,
            inp,
            graph_channel=graph_channel,
        )
        if remote_push is not None:
            threads, blocks, stream_mode = remote_push
            transport = "tp2_remote_push_stream" if stream_mode else "tp2_remote_push"
            return transport, threads, blocks
        threads, blocks = cls._launch_geometry(inp.numel() * inp.element_size() // 16)
        return "pull", threads, blocks

    @classmethod
    def _plain_graph_plan(
        cls,
        state: _CuTeOneshotState,
        inp: torch.Tensor,
    ) -> _PlainGraphLaunchPlan:
        """Create the immutable launch contract admitted before graph capture."""

        transport, threads, blocks = cls._plain_launch_config(
            state,
            inp,
            graph_channel=True,
        )
        slot_ptrs = (0, 0, 0, 0)
        if transport.startswith("tp2_remote_push"):
            if state.eager_ptrs is None:
                raise RuntimeError("TP2 peer-push requires eager staging pointers")
            peer_rank = state.rank ^ 1
            slot_ptrs = (
                state.eager_ptrs[0][state.rank],
                state.eager_ptrs[0][peer_rank],
                state.eager_ptrs[1][state.rank],
                state.eager_ptrs[1][peer_rank],
            )
        return _PlainGraphLaunchPlan(
            transport=transport,
            threads=threads,
            blocks=blocks,
            size_packs=inp.numel() * inp.element_size() // 16,
            stage_input=state.eager_tables is not None,
            device_index=cls._device_index(inp.device),
            eager_buffer_bytes=state.eager_buffer_bytes,
            transport_policy=state.transport_policy,
            remote_push_region_packs=state.plain_remote_push_region_packs,
            remote_push_slot_ptrs=slot_ptrs,
        )

    @classmethod
    def _plain_graph_plan_key(
        cls,
        inp: torch.Tensor,
    ) -> tuple[object, ...]:
        """Return the tensor contract that identifies a prepared graph plan."""

        return (
            _dtype_name(inp.dtype),
            tuple(inp.shape),
            tuple(inp.stride()),
            cls._device_index(inp.device),
        )

    @staticmethod
    def _fused_threads() -> int:
        threads = int(os.getenv("B12X_PCIE_FUSED_THREADS", "256"))
        return min(512, max(64, (threads // 32) * 32))

    @staticmethod
    def _fused_topology_mode(
        state: _CuTeOneshotState,
        inp: torch.Tensor,
    ) -> Optional[str]:
        """Select only topology transports with a qualified shape contract."""

        if (
            state.eager_tables is None
            or state.eager_buffer_bytes is None
            or inp.dtype not in (torch.float16, torch.bfloat16)
            or inp.ndim == 0
        ):
            return None
        hidden = int(inp.shape[-1])
        if hidden <= 0 or inp.numel() % hidden != 0:
            return None
        rows = inp.numel() // hidden
        _, tp2_remote, _, tp4_remote, tp8_owner = state.transport_policy
        if (
            state.world_size == 8
            and tp8_owner
            and hidden in (4096, 6144)
            and 2 <= rows <= 8
        ):
            return "stage_tp8_owner"
        if (
            state.world_size == 4
            and tp4_remote
            and hidden in (4096, 6144)
            and 1 <= rows <= 32
        ):
            return "stage_remote_push"
        if state.world_size == 2 and tp2_remote and hidden == 4096 and 1 <= rows <= 32:
            return "stage_remote_push"
        return None

    @classmethod
    def _fused_launch_config(
        cls,
        state: _CuTeOneshotState,
        inp: torch.Tensor,
    ) -> tuple[str, bool, bool, int]:
        pack_elems = 16 // inp.element_size()
        hidden_packs = int(inp.shape[-1]) // pack_elems
        rows = inp.numel() // int(inp.shape[-1])
        threads = cls._fused_threads()
        override = int(os.getenv("B12X_PCIE_FUSED_CTAS_PER_ROW", "0"))
        if override > 0:
            ctas_per_row = override
        else:
            min_ctas = (hidden_packs + threads * _REG_PACKS - 1) // (
                threads * _REG_PACKS
            )
            ctas_per_row = max(max(1, 3 // rows), min_ctas)
        ctas_per_row = max(1, min(ctas_per_row, _MAX_BLOCKS // rows))
        register_normalize = (hidden_packs + ctas_per_row * threads - 1) // (
            ctas_per_row * threads
        ) <= _REG_PACKS
        topology_mode = cls._fused_topology_mode(state, inp)
        if topology_mode is not None:
            mode = topology_mode
        elif state.eager_tables is None:
            mode = "registered"
        elif state.transport_policy[0]:
            mode = "stage_push"
        else:
            mode = "stage_pull"
        return mode, ctas_per_row == 1, register_normalize, threads

    @staticmethod
    def _device_index(device: torch.device) -> int:
        return device.index if device.index is not None else torch.cuda.current_device()

    def _prepare_all_reduce(self, handle: int, inp: torch.Tensor) -> None:
        """Compile/load the eager and every graph slot variant without launching."""

        state = self._state(handle)
        stage_input = state.eager_tables is not None
        if not stage_input and int(inp.data_ptr()) not in state.registered_tables:
            raise RuntimeError("input buffer is not registered")
        plan = self._plain_graph_plan(state, inp)
        eager_transport, eager_threads, _ = self._plain_launch_config(state, inp)
        from ._oneshot_cute import get_oneshot_launcher

        variants = (
            ((False, 0), (True, 0), (True, 1))
            if stage_input
            else ((False, 0),)
        )
        for device_slot_selection, slot_bias in variants:
            get_oneshot_launcher(
                _dtype_name(inp.dtype),
                state.world_size,
                state.rank,
                stage_input,
                device_slot_selection,
                slot_bias,
                plan.transport if device_slot_selection else eager_transport,
                plan.threads if device_slot_selection else eager_threads,
                plan.device_index,
            )
        state.plain_graph_plans[self._plain_graph_plan_key(inp)] = plan

    def _prepare_fused_all_reduce(self, handle: int, inp: torch.Tensor) -> None:
        """Compile/load the eager and every fused graph variant without launching."""

        state = self._state(handle)
        if (
            state.eager_tables is None
            and int(inp.data_ptr()) not in state.registered_tables
        ):
            raise RuntimeError("input buffer is not registered")
        mode, single_cta, register_normalize, threads = self._fused_launch_config(
            state, inp
        )
        from ._oneshot_cute import get_fused_oneshot_launcher

        device_index = self._device_index(inp.device)
        variants = (
            ((False, 0), (True, 0), (True, 1))
            if state.eager_tables is not None
            else ((False, 0),)
        )
        for device_slot_selection, slot_bias in variants:
            get_fused_oneshot_launcher(
                _dtype_name(inp.dtype),
                state.world_size,
                state.rank,
                mode,
                single_cta,
                register_normalize,
                device_slot_selection,
                slot_bias,
                threads,
                device_index,
            )

    @staticmethod
    def _select_table(state: _CuTeOneshotState, input_pointer: int) -> tuple[int, bool]:
        if state.eager_tables is not None:
            if state.device_slot_selection:
                # The two 16-entry tables are adjacent in rank_data.  A
                # graph-safe kernel selects between them from channel-local
                # device state without another peer barrier.
                return state.eager_tables[0], True
            slot = state.eager_slot % 2
            state.eager_slot += 1
            return state.eager_tables[slot], True
        try:
            return state.registered_tables[int(input_pointer)], False
        except KeyError as exc:
            raise RuntimeError(
                f"buffer address {int(input_pointer)} is not registered"
            ) from exc

    def all_reduce(
        self,
        handle: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        _reg_buffer: int,
        _reg_buffer_sz_bytes: int,
    ) -> None:
        del _reg_buffer, _reg_buffer_sz_bytes
        state = self._state(handle)
        capturing = _is_current_stream_capturing(inp.device)
        size_packs = inp.numel() * inp.element_size() // 16
        graph_channel = state.device_slot_selection or capturing
        plan_key = self._plain_graph_plan_key(inp)
        plan = state.plain_graph_plans.get(plan_key) if graph_channel else None
        bind_plan_after_launcher_check = False
        if graph_channel and plan is None:
            if capturing:
                # PIECEWISE capture can expose a row count outside the explicit
                # capture-size set. Binding host metadata is safe only when the
                # required launcher specialization is already resident.
                plan = self._plain_graph_plan(state, inp)
                bind_plan_after_launcher_check = True
            else:
                self._prepare_all_reduce(handle, inp)
                plan = state.plain_graph_plans[plan_key]
        if plan is None:
            transport, threads, blocks = self._plain_launch_config(
                state,
                inp,
                graph_channel=False,
            )
            device_index = self._device_index(inp.device)
            plain_region_packs = 0
            plain_capacity_packs = 0
            plain_slot_ptrs = (0, 0, 0, 0)
        else:
            transport = plan.transport
            threads = plan.threads
            blocks = plan.blocks
            size_packs = plan.size_packs
            device_index = plan.device_index
            plain_region_packs = plan.remote_push_region_packs
            plain_capacity_packs = int(plan.eager_buffer_bytes or 0) // 16
            plain_slot_ptrs = plan.remote_push_slot_ptrs
        prospective_device_selection = state.device_slot_selection or (
            capturing and state.eager_tables is not None
        )
        prospective_slot_bias = (
            state.slot_bias if state.device_slot_selection else state.eager_slot & 1
        )
        if capturing:
            from ._oneshot_cute import is_oneshot_launcher_prepared

            if not is_oneshot_launcher_prepared(
                _dtype_name(inp.dtype),
                state.world_size,
                state.rank,
                state.eager_tables is not None,
                prospective_device_selection,
                prospective_slot_bias,
                transport,
                threads,
                device_index,
            ):
                raise RuntimeError(
                    "cold PCIe oneshot CUDA graph capture is not allowed; "
                    "call prepare_graph_all_reduce() before capture"
                )
            if bind_plan_after_launcher_check:
                state.plain_graph_plans[plan_key] = plan
        if (
            capturing
            and state.eager_tables is not None
            and not state.device_slot_selection
        ):
            # The graph epoch begins at zero.  Bias it to the host slot that
            # would have been selected for this invocation, then keep both
            # values fixed in the captured specialization.
            _enable_device_slot_selection(state, capturing=True)
        table_address, stage_input = self._select_table(state, inp.data_ptr())
        from ._oneshot_cute import get_oneshot_launcher

        with torch.cuda.device(inp.device):
            launcher = get_oneshot_launcher(
                _dtype_name(inp.dtype),
                state.world_size,
                state.rank,
                stage_input,
                state.device_slot_selection,
                state.slot_bias,
                transport,
                threads,
                device_index,
            )
            launcher(
                table_address,
                state.signal_table_address,
                inp.data_ptr(),
                out.data_ptr(),
                size_packs,
                plain_region_packs,
                plain_capacity_packs,
                *plain_slot_ptrs,
                blocks,
            )

    def all_reduce_fused_add_rms_norm(
        self,
        handle: int,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        residual_out: torch.Tensor,
        epsilon: float,
        _reg_buffer: int,
        _reg_buffer_sz_bytes: int,
    ) -> None:
        del _reg_buffer, _reg_buffer_sz_bytes
        state = self._state(handle)
        capturing = _is_current_stream_capturing(inp.device)
        mode, single_cta, register_normalize, threads = self._fused_launch_config(
            state, inp
        )
        device_index = self._device_index(inp.device)
        prospective_device_selection = state.device_slot_selection or (
            capturing and state.eager_tables is not None
        )
        prospective_slot_bias = (
            state.slot_bias if state.device_slot_selection else state.eager_slot & 1
        )
        if capturing:
            from ._oneshot_cute import is_fused_oneshot_launcher_prepared

            if not is_fused_oneshot_launcher_prepared(
                _dtype_name(inp.dtype),
                state.world_size,
                state.rank,
                mode,
                single_cta,
                register_normalize,
                prospective_device_selection,
                prospective_slot_bias,
                threads,
                device_index,
            ):
                raise RuntimeError(
                    "cold fused PCIe oneshot CUDA graph capture is not allowed; "
                    "call prepare_graph_fused_add_rms_norm() before capture"
                )
        _enable_device_slot_selection(state, capturing=capturing)
        table_address, staged = self._select_table(state, inp.data_ptr())
        pack_elems = 16 // inp.element_size()
        hidden_packs = int(inp.shape[-1]) // pack_elems
        rows = inp.numel() // int(inp.shape[-1])
        if rows > _MAX_BLOCKS:
            raise ValueError(
                f"fused allreduce RMSNorm supports at most {_MAX_BLOCKS} rows"
            )
        _, single_cta_check, _, _ = self._fused_launch_config(state, inp)
        # Recover the CTA count from the already-derived single-CTA/config
        # formula without changing the launch contract.
        override = int(os.getenv("B12X_PCIE_FUSED_CTAS_PER_ROW", "0"))
        if override > 0:
            ctas_per_row = override
        else:
            min_ctas = (hidden_packs + threads * _REG_PACKS - 1) // (
                threads * _REG_PACKS
            )
            ctas_per_row = max(max(1, 3 // rows), min_ctas)
        ctas_per_row = max(1, min(ctas_per_row, _MAX_BLOCKS // rows))
        assert single_cta_check == (ctas_per_row == 1)
        blocks = rows * ctas_per_row

        from ._oneshot_cute import get_fused_oneshot_launcher

        with torch.cuda.device(inp.device):
            launcher = get_fused_oneshot_launcher(
                _dtype_name(inp.dtype),
                state.world_size,
                state.rank,
                mode,
                single_cta,
                register_normalize,
                state.device_slot_selection,
                state.slot_bias,
                threads,
                device_index,
            )
            if not state.device_slot_selection and staged:
                for slot_bias in (0, 1):
                    get_fused_oneshot_launcher(
                        _dtype_name(inp.dtype),
                        state.world_size,
                        state.rank,
                        mode,
                        single_cta,
                        register_normalize,
                        True,
                        slot_bias,
                        threads,
                        device_index,
                    )
            launcher(
                table_address,
                state.signal_table_address,
                inp.data_ptr(),
                residual.data_ptr(),
                weight.data_ptr(),
                out.data_ptr(),
                residual_out.data_ptr(),
                hidden_packs,
                rows,
                ctas_per_row,
                int(state.eager_buffer_bytes or inp.numel() * inp.element_size()) // 16,
                float(epsilon),
                blocks,
            )

    def get_graph_buffer_ipc_meta(self, handle: int):
        self._state(handle)
        # The native implementation never populated its graph-unregistered
        # list. Eager IPC buffers are the supported graph path in both runtimes.
        return [], []

    def register_graph_buffers(self, handle: int, handles, offsets) -> None:
        self._state(handle)
        if any(handles) or any(offsets):
            raise RuntimeError("CuTe oneshot graph capture requires eager IPC buffers")

    def dispose(self, handle: int) -> None:
        with self._lock:
            self._states.pop(int(handle), None)


def _compute_crossover_size(
    benchmark: Callable[[int], tuple[float, float]],
    *,
    ceiling_bytes: int = AUTOTUNE_CEILING,
    fine_step_bytes: int = AUTOTUNE_FINE_STEP,
) -> tuple[int, list[_BenchmarkResult]]:
    coarse_sizes = []
    current = 1024
    while current <= ceiling_bytes:
        coarse_sizes.append(current)
        current *= 2

    results: list[_BenchmarkResult] = []
    seen_sizes: set[int] = set()
    first_nccl_win: Optional[int] = None
    last_custom_win = 0

    def record(size_bytes: int) -> None:
        nonlocal first_nccl_win, last_custom_win
        custom_us, nccl_us = benchmark(size_bytes)
        winner = "custom" if custom_us < nccl_us else "NCCL"
        results.append(_BenchmarkResult(size_bytes, custom_us, nccl_us, winner))
        seen_sizes.add(size_bytes)
        if winner == "custom":
            last_custom_win = max(last_custom_win, size_bytes)
        elif first_nccl_win is None:
            first_nccl_win = size_bytes

    for size_bytes in coarse_sizes:
        record(size_bytes)

    if last_custom_win > 0 and first_nccl_win is not None:
        fine_start = last_custom_win
        fine_end = min(first_nccl_win, last_custom_win * 4)
        fine_size = fine_start + fine_step_bytes
        while fine_size < fine_end:
            aligned = (fine_size // 16) * 16
            if aligned not in seen_sizes:
                record(aligned)
            fine_size += fine_step_bytes

    results.sort(key=lambda item: item.size_bytes)
    crossover = 1024
    for result in results:
        if result.winner == "custom":
            crossover = result.size_bytes
    return crossover, results


class PCIeOneshotAllReduce:
    """Standalone unfused PCIe oneshot allreduce runtime."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]] = None,
        eager_buffer_ptrs1: Optional[Sequence[int]] = None,
        eager_buffer_bytes: Optional[int] = None,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ):
        resolved_group = _resolve_exchange_group(exchange_group, process_group)

        def normalize_and_validate():
            device_obj = _normalize_device(device)
            normalized_rank = int(rank)
            normalized_world_size = int(world_size)
            normalized_signals = tuple(int(ptr) for ptr in signal_ptrs)
            normalized_eager0 = (
                None
                if eager_buffer_ptrs0 is None
                else tuple(int(ptr) for ptr in eager_buffer_ptrs0)
            )
            normalized_eager1 = (
                None
                if eager_buffer_ptrs1 is None
                else tuple(int(ptr) for ptr in eager_buffer_ptrs1)
            )
            normalized_max_size = int(max_size)
            normalized_rank_data_bytes = int(rank_data_bytes)
            normalized_eager_bytes = (
                None if eager_buffer_bytes is None else int(eager_buffer_bytes)
            )
            normalized_transport_policy = _transport_policy_contract()
            if normalized_world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {normalized_world_size}")
            if not 0 <= normalized_rank < normalized_world_size:
                raise ValueError(
                    f"invalid rank {normalized_rank} for world size "
                    f"{normalized_world_size}"
                )
            if len(normalized_signals) != normalized_world_size:
                raise ValueError("signal_ptrs must match world size")
            if (normalized_eager0 is None) != (normalized_eager1 is None):
                raise ValueError("eager buffers must be provided as a pair")
            if (
                normalized_eager0 is not None
                and len(normalized_eager0) != normalized_world_size
            ):
                raise ValueError("eager_buffer_ptrs0 must match world size")
            if (
                normalized_eager1 is not None
                and len(normalized_eager1) != normalized_world_size
            ):
                raise ValueError("eager_buffer_ptrs1 must match world size")
            if normalized_max_size <= 0:
                raise ValueError("max_size must be positive")
            if normalized_rank_data_bytes <= 0:
                raise ValueError("rank_data_bytes must be positive")
            if normalized_eager0 is not None:
                if normalized_eager_bytes is None:
                    raise ValueError(
                        "eager_buffer_bytes is required with eager buffer pointers"
                    )
                if normalized_eager_bytes <= 0:
                    raise ValueError("eager_buffer_bytes must be positive")
                if normalized_max_size > normalized_eager_bytes:
                    raise ValueError("max_size exceeds eager buffer capacity")
            if ext_module is None and device_obj.type != "cuda":
                raise ValueError("PCIe oneshot allreduce requires a CUDA device")
            if device_obj.type == "cuda" and resolved_group is None:
                raise ValueError(
                    "exchange_group is required for a CUDA PCIe oneshot runtime; "
                    "use from_exchange_group()"
                )
            if resolved_group is not None:
                if device_obj.type != "cuda":
                    raise ValueError(
                        "distributed PCIe oneshot allreduce requires a CUDA device"
                    )
                group_rank = dist.get_rank(group=resolved_group)
                group_world_size = dist.get_world_size(group=resolved_group)
                if normalized_rank != group_rank:
                    raise ValueError(
                        f"supplied rank {normalized_rank} does not match process "
                        f"group rank {group_rank}"
                    )
                if normalized_world_size != group_world_size:
                    raise ValueError(
                        f"supplied world size {normalized_world_size} does not "
                        f"match process group size {group_world_size}"
                    )
            return (
                device_obj,
                normalized_rank,
                normalized_world_size,
                normalized_signals,
                normalized_eager0,
                normalized_eager1,
                normalized_eager_bytes,
                normalized_transport_policy,
                normalized_max_size,
                normalized_rank_data_bytes,
            )

        if resolved_group is not None:
            normalized = _run_collective_preallocation_setup(
                owner="PCIe oneshot direct constructor argument validation",
                exchange_group=resolved_group,
                setup=normalize_and_validate,
            )
        else:
            normalized = normalize_and_validate()

        (
            device_obj,
            normalized_rank,
            normalized_world_size,
            normalized_signals,
            normalized_eager0,
            normalized_eager1,
            normalized_eager_bytes,
            normalized_transport_policy,
            normalized_max_size,
            normalized_rank_data_bytes,
        ) = normalized

        self._initialize_prepared_state(
            rank=normalized_rank,
            world_size=normalized_world_size,
            device=device_obj,
            signal_ptrs=normalized_signals,
            eager_buffer_ptrs0=normalized_eager0,
            eager_buffer_ptrs1=normalized_eager1,
            eager_buffer_bytes=normalized_eager_bytes,
            transport_policy=normalized_transport_policy,
            exchange_group=resolved_group,
            ipc=ipc,
            owned_buffers=owned_buffers,
            max_size=normalized_max_size,
            rank_data_bytes=normalized_rank_data_bytes,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

        if self.device.type == "cuda":
            _require_full_grid_residency(
                owner="PCIe oneshot direct constructor",
                required_sms=ONESHOT_REQUIRED_SMS,
                device=self.device,
                exchange_group=self.exchange_group,
            )

            def prepare() -> tuple[Optional[CudaRTLibrary], object]:
                prepared_ipc = self._ipc
                if prepared_ipc is None and (ext_module is None or owned_buffers):
                    prepared_ipc = CudaRTLibrary()
                if prepared_ipc is not None:
                    prepared_ipc.cudaSetDevice(_cuda_device_index(self.device))
                return prepared_ipc, ext_module or _load_extension()

            if self.exchange_group is None:
                self._ipc, self._ext = prepare()
            else:
                self._ipc, self._ext = _run_collective_preallocation_setup(
                    owner="PCIe oneshot direct constructor",
                    exchange_group=self.exchange_group,
                    setup=prepare,
                )
        else:
            self._ext = self._ext or _load_extension()

        if self.device.type == "cuda" and self.exchange_group is not None:
            _require_collective_contract(
                owner="PCIe oneshot direct constructor",
                exchange_group=self.exchange_group,
                contract=(
                    self.world_size,
                    self.max_size,
                    self.rank_data_bytes,
                    self.eager_buffer_bytes,
                    self._pending_eager_ptrs is not None,
                    self._transport_policy,
                ),
            )

        init_error = self._initialize_native_runtime()
        if self.device.type == "cuda" and self.exchange_group is not None:

            def abort_native_runtime() -> None:
                if self._ptr:
                    self._ext.dispose(self._ptr)
                    self._ptr = 0

            _finish_collective_unowned_runtime_setup(
                owner="PCIe oneshot direct constructor",
                exchange_group=self.exchange_group,
                local_error=init_error,
                local_cleanup=abort_native_runtime,
            )
        elif init_error is not None:
            raise init_error

    def _initialize_prepared_state(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]],
        eager_buffer_ptrs1: Optional[Sequence[int]],
        eager_buffer_bytes: Optional[int],
        transport_policy: tuple[bool, bool, bool, bool, bool],
        exchange_group: Optional[ProcessGroup],
        ipc: Optional[CudaRTLibrary],
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]],
        max_size: int,
        rank_data_bytes: int,
        ext_module,
        stream_affine: bool,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.exchange_group = exchange_group
        self.process_group = exchange_group
        self.max_size = int(max_size)
        self.rank_data_bytes = int(rank_data_bytes)
        self.eager_buffer_bytes = (
            None if eager_buffer_bytes is None else int(eager_buffer_bytes)
        )
        self._transport_policy = tuple(bool(value) for value in transport_policy)
        self._sharded_eager_storage = _uses_sharded_eager_storage(
            self.world_size,
            self._transport_policy,
        )
        self._ipc = ipc
        self._ext = ext_module
        self._signal_ptrs = tuple(int(ptr) for ptr in signal_ptrs)
        self._pending_eager_ptrs = (
            None
            if eager_buffer_ptrs0 is None or eager_buffer_ptrs1 is None
            else (
                tuple(int(ptr) for ptr in eager_buffer_ptrs0),
                tuple(int(ptr) for ptr in eager_buffer_ptrs1),
            )
        )
        self._owned_buffers = list(owned_buffers or [])
        self._registered_input_ptrs: dict[int, tuple[int, ...]] = {}
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._ipc_imports_closed = False
        self._ipc_exports_freed = False
        self._coordinated_close_complete = False
        self._closed_ipc_import_indices: set[tuple[int, int]] = set()
        self._ptr = 0
        self._eager_ptrs: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None

    def _initialize_native_runtime(self) -> BaseException | None:
        init_error: BaseException | None = None
        try:
            self.rank_data = torch.empty(
                self.rank_data_bytes, dtype=torch.uint8, device=self.device
            )
            self._ptr = self._ext.init_custom_ar(
                list(self._signal_ptrs), self.rank_data, self.rank
            )
            if self._pending_eager_ptrs is not None:
                self._eager_ptrs = self._pending_eager_ptrs
                if getattr(
                    self._ext,
                    "supports_eager_storage_contract",
                    False,
                ):
                    self._ext.register_pcie_buffers(
                        self._ptr,
                        list(self._eager_ptrs[0]),
                        list(self._eager_ptrs[1]),
                        self.eager_buffer_bytes,
                        self._transport_policy,
                    )
                else:
                    self._ext.register_pcie_buffers(
                        self._ptr,
                        list(self._eager_ptrs[0]),
                        list(self._eager_ptrs[1]),
                    )
        except Exception as exc:
            init_error = exc
        finally:
            self._pending_eager_ptrs = None
        return init_error

    @classmethod
    def _from_prepared_factory(
        cls,
        **kwargs,
    ) -> tuple["PCIeOneshotAllReduce", BaseException | None]:
        runtime = object.__new__(cls)
        runtime._initialize_prepared_state(**kwargs)
        init_error = runtime._initialize_native_runtime()
        return runtime, init_error

    @classmethod
    def from_ipc(
        cls,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]] = None,
        eager_buffer_ptrs1: Optional[Sequence[int]] = None,
        eager_buffer_bytes: Optional[int] = None,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeOneshotAllReduce":
        return cls(
            rank=rank,
            world_size=world_size,
            device=device,
            signal_ptrs=signal_ptrs,
            eager_buffer_ptrs0=eager_buffer_ptrs0,
            eager_buffer_ptrs1=eager_buffer_ptrs1,
            eager_buffer_bytes=eager_buffer_bytes,
            exchange_group=exchange_group,
            process_group=process_group,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeOneshotAllReduce":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)

        def validate_factory_arguments():
            device_obj = _normalize_device(device)
            normalized_eager_bytes = int(eager_buffer_bytes)
            normalized_max_size = int(max_size)
            normalized_rank_data_bytes = int(rank_data_bytes)
            normalized_transport_policy = _transport_policy_contract()
            if world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {world_size}")
            if device_obj.type != "cuda":
                raise ValueError("PCIe oneshot requires a CUDA device")
            if normalized_eager_bytes <= 0:
                raise ValueError("eager_buffer_bytes must be positive")
            if normalized_max_size <= 0:
                raise ValueError("max_size must be positive")
            if normalized_max_size > normalized_eager_bytes:
                raise ValueError("max_size exceeds eager buffer capacity")
            if normalized_rank_data_bytes <= 0:
                raise ValueError("rank_data_bytes must be positive")
            return (
                device_obj,
                normalized_eager_bytes,
                normalized_max_size,
                normalized_rank_data_bytes,
                normalized_transport_policy,
            )

        (
            device_obj,
            eager_buffer_bytes,
            max_size,
            rank_data_bytes,
            transport_policy,
        ) = _run_collective_preallocation_setup(
            owner="PCIe oneshot argument validation",
            exchange_group=exchange_group,
            setup=validate_factory_arguments,
        )

        _require_full_grid_residency(
            owner="PCIe oneshot",
            required_sms=ONESHOT_REQUIRED_SMS,
            device=device_obj,
            exchange_group=exchange_group,
        )

        def prepare() -> tuple[CudaRTLibrary, object, int]:
            prepared_ipc = CudaRTLibrary()
            prepared_ipc.cudaSetDevice(_cuda_device_index(device_obj))
            prepared_ext = ext_module or _load_extension()
            return prepared_ipc, prepared_ext, int(prepared_ext.meta_size())

        ipc, ext, signal_bytes = _run_collective_preallocation_setup(
            owner="PCIe oneshot",
            exchange_group=exchange_group,
            setup=prepare,
        )
        _require_collective_contract(
            owner="PCIe oneshot channel layout",
            exchange_group=exchange_group,
            contract=(
                signal_bytes,
                eager_buffer_bytes,
                max_size,
                rank_data_bytes,
                transport_policy,
            ),
        )

        channel_buffers = cls._allocate_eager_channel_buffers(
            exchange_group,
            signal_bytes=signal_bytes,
            eager_buffer_bytes=eager_buffer_bytes,
            eager_storage_shards=_eager_storage_shards(
                world_size,
                transport_policy,
            ),
            ipc=ipc,
        )
        owned_buffers = [channel_buffers.owned_buffer]
        runtime, init_error = cls._from_prepared_factory(
            rank=rank,
            world_size=world_size,
            device=device_obj,
            signal_ptrs=channel_buffers.signal_ptrs,
            eager_buffer_ptrs0=channel_buffers.eager0_ptrs,
            eager_buffer_ptrs1=channel_buffers.eager1_ptrs,
            eager_buffer_bytes=eager_buffer_bytes,
            transport_policy=transport_policy,
            exchange_group=exchange_group,
            ipc=ipc,
            owned_buffers=owned_buffers,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext,
            stream_affine=stream_affine,
        )

        def abort_native_runtime() -> None:
            pointer = getattr(runtime, "_ptr", 0)
            if pointer:
                ext.dispose(pointer)
                runtime._ptr = 0
            if hasattr(runtime, "rank_data"):
                del runtime.rank_data

        def detach_shared_ownership() -> None:
            runtime._owned_buffers.clear()

        _finish_collective_runtime_setup(
            owner="PCIe oneshot",
            exchange_group=exchange_group,
            ipc=ipc,
            shared=channel_buffers.owned_buffer,
            local_error=init_error,
            local_cleanup=abort_native_runtime,
            detach_shared_ownership=detach_shared_ownership,
        )
        return runtime

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: Optional[int] = None,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeOneshotAllReduce":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=max_input_bytes
            if eager_buffer_bytes is None
            else eager_buffer_bytes,
            max_size=max_input_bytes if max_size is None else max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

    @staticmethod
    def _allocate_shared_buffer(
        exchange_group: ProcessGroup,
        size_in_bytes: int,
        *,
        zero_fill: bool,
        ipc: CudaRTLibrary,
        peer_ranks: Optional[Sequence[int]] = None,
    ) -> _OwnedSharedBuffer:
        _require_no_retained_ipc_setup(exchange_group)
        world_size = dist.get_world_size(group=exchange_group)
        rank = dist.get_rank(group=exchange_group)
        selected_peers = (
            set(range(world_size)) - {rank}
            if peer_ranks is None
            else {int(peer) for peer in peer_ranks}
        )
        if rank in selected_peers or any(
            not 0 <= peer < world_size for peer in selected_peers
        ):
            raise ValueError("peer_ranks must contain valid remote group ranks")

        local_ptr: int | None = None
        local_handle: bytes | None = None
        prepare_error: BaseException | None = None
        try:
            local_ptr = ipc.cudaMalloc(size_in_bytes)
            if zero_fill:
                ipc.cudaMemset(local_ptr, 0, size_in_bytes)
            local_handle = ipc.cudaIpcGetMemHandleBytes(local_ptr)
        except Exception as exc:
            prepare_error = exc

        # A rank that cannot allocate or publish its export must not leave its
        # peers blocked in the handle exchange.  No imports exist yet, so
        # successful ranks can safely release their local allocation.
        try:
            prepare_statuses = _exchange_setup_failures(
                prepare_error,
                exchange_group=exchange_group,
                phase="CUDA IPC export preparation",
            )
        except Exception as exchange_error:
            retained = _retain_failed_ipc_setup(
                ipc=ipc,
                local_ptr=int(local_ptr or 0),
                exchange_group=exchange_group,
                owner="PCIe shared buffer",
                phase="CUDA IPC export preparation",
                state="unmap",
            )
            error = RuntimeError(
                "failed to exchange CUDA IPC export preparation status; "
                "CUDA IPC ownership was retained"
            )
            raise _attach_retryable_setup(error, retained) from (
                prepare_error or exchange_error
            )
        if any(prepare_statuses):
            free_error: BaseException | None = None
            live_local_ptr = int(local_ptr or 0)
            if local_ptr is not None:
                try:
                    ipc.cudaFree(local_ptr)
                except Exception as exc:
                    free_error = exc
                else:
                    live_local_ptr = 0
            try:
                free_statuses = _exchange_setup_failures(
                    free_error,
                    exchange_group=exchange_group,
                    phase="CUDA IPC export preparation rollback",
                )
            except Exception as exchange_error:
                retained_export = _retain_failed_ipc_setup(
                    ipc=ipc,
                    local_ptr=live_local_ptr,
                    exchange_group=exchange_group,
                    owner="PCIe shared buffer",
                    phase="CUDA IPC export preparation",
                    state="free",
                )
                error = RuntimeError(
                    "failed to exchange CUDA IPC export rollback status; "
                    "CUDA IPC ownership was retained"
                )
                raise _attach_retryable_setup(error, retained_export) from (
                    free_error or prepare_error or exchange_error
                )
            if any(free_statuses):
                retained_export = _retain_failed_ipc_setup(
                    ipc=ipc,
                    local_ptr=live_local_ptr,
                    exchange_group=exchange_group,
                    owner="PCIe shared buffer",
                    phase="CUDA IPC export preparation",
                    state="free",
                )
                error = RuntimeError(
                    _setup_failure_message(
                        "PCIe shared buffer",
                        "export preparation rollback",
                        free_statuses,
                        exports_retained=True,
                    )
                )
                raise _attach_retryable_setup(error, retained_export) from (
                    prepare_error or free_error
                )
            raise RuntimeError(
                _setup_failure_message(
                    "PCIe shared buffer",
                    "export preparation",
                    prepare_statuses,
                    exports_retained=False,
                )
            ) from prepare_error

        assert local_ptr is not None
        assert local_handle is not None
        peer_ptrs: list[int] = []
        remote_ptrs: list[int] = []
        try:
            handles = _broadcast_gather_object(local_handle, exchange_group)
        except Exception as exchange_error:
            # No rank has opened an import before handle exchange completes.
            # Retain both ownership and a collective retry ticket if progress
            # is uncertain; peers may have completed the exchange already.
            retained = _retain_failed_ipc_setup(
                ipc=ipc,
                local_ptr=local_ptr,
                exchange_group=exchange_group,
                owner="PCIe shared buffer",
                phase="CUDA IPC handle exchange",
                state="unmap",
            )
            error = RuntimeError(
                "CUDA IPC handle exchange failed; CUDA IPC ownership was retained"
            )
            raise _attach_retryable_setup(error, retained) from exchange_error

        open_error: BaseException | None = None
        for idx, handle in enumerate(handles):
            if idx == rank:
                peer_ptrs.append(local_ptr)
                continue
            if idx not in selected_peers:
                peer_ptrs.append(0)
                continue
            try:
                remote_ptr = ipc.cudaIpcOpenMemHandleBytes(handle)
            except Exception as exc:
                if open_error is None:
                    open_error = RuntimeError(
                        f"failed to open CUDA IPC handle for peer group rank {idx}"
                    )
                    open_error.__cause__ = exc
                # Preserve group-rank indexing while every rank completes the
                # same open loop and reaches the collective verdict.
                peer_ptrs.append(0)
            else:
                peer_ptrs.append(remote_ptr)
                remote_ptrs.append(remote_ptr)

        if len(handles) != world_size or len(peer_ptrs) != world_size:
            open_error = open_error or RuntimeError(
                "failed to gather CUDA IPC handles for every group rank"
            )
        try:
            open_statuses = _exchange_setup_failures(
                open_error,
                exchange_group=exchange_group,
                phase="CUDA IPC import open",
            )
        except Exception as exchange_error:
            retained = _retain_failed_ipc_setup(
                ipc=ipc,
                local_ptr=local_ptr,
                exchange_group=exchange_group,
                owner="PCIe shared buffer",
                phase="CUDA IPC import open",
                remote_ptrs=remote_ptrs,
                state="unmap",
            )
            error = RuntimeError(
                "failed to exchange CUDA IPC import-open status; CUDA IPC "
                "ownership was retained"
            )
            raise _attach_retryable_setup(error, retained) from (
                open_error or exchange_error
            )
        if any(open_statuses):
            _abort_collective_ipc_setup(
                owner="PCIe shared buffer",
                setup_phase="CUDA IPC import open",
                setup_statuses=open_statuses,
                exchange_group=exchange_group,
                ipc=ipc,
                local_ptr=local_ptr,
                remote_ptrs=remote_ptrs,
                local_error=open_error,
            )

        return _OwnedSharedBuffer(
            local_ptr=local_ptr,
            peer_ptrs=tuple(peer_ptrs),
            remote_ptrs=tuple(remote_ptrs),
        )

    @classmethod
    def _allocate_eager_channel_buffers(
        cls,
        exchange_group: ProcessGroup,
        *,
        signal_bytes: int,
        eager_buffer_bytes: int,
        eager_storage_shards: int,
        ipc: CudaRTLibrary,
    ) -> _ChannelSharedBuffers:
        signal_bytes = int(signal_bytes)
        eager_buffer_bytes = int(eager_buffer_bytes)
        if signal_bytes <= 0:
            raise ValueError("signal_bytes must be positive")
        if eager_buffer_bytes <= 0:
            raise ValueError("eager_buffer_bytes must be positive")
        eager_storage_shards = int(eager_storage_shards)
        if eager_storage_shards <= 0:
            raise ValueError("eager_storage_shards must be positive")
        eager_buffer_bytes *= eager_storage_shards

        signal_offset = 0
        eager0_offset = _align_up(signal_bytes, IPC_SLAB_ALIGNMENT)
        eager1_offset = eager0_offset + _align_up(
            eager_buffer_bytes, IPC_SLAB_ALIGNMENT
        )
        slab_bytes = eager1_offset + eager_buffer_bytes
        slab = cls._allocate_shared_buffer(
            exchange_group,
            slab_bytes,
            # Clear signals before publishing the IPC handle so a peer cannot
            # post an arrival that a later local memset would erase.
            zero_fill=True,
            ipc=ipc,
        )
        return _ChannelSharedBuffers(
            owned_buffer=slab,
            signal_ptrs=tuple(ptr + signal_offset for ptr in slab.peer_ptrs),
            eager0_ptrs=tuple(ptr + eager0_offset for ptr in slab.peer_ptrs),
            eager1_ptrs=tuple(ptr + eager1_offset for ptr in slab.peer_ptrs),
        )

    @property
    def signal_ptrs(self) -> tuple[int, ...]:
        return self._signal_ptrs

    def _bind_stream_key(self, stream_key: Optional[int]) -> None:
        if not self._stream_affine:
            return
        if stream_key is None:
            return
        if self._owner_stream_key is None:
            self._owner_stream_key = int(stream_key)
            return
        if self._owner_stream_key != int(stream_key):
            raise RuntimeError(
                "PCIe oneshot allreduce channels are stream-affine; "
                "create or use a separate channel for each CUDA stream"
            )

    def _check_stream(self, stream: object = None) -> None:
        if stream is None and _is_current_stream_capturing(self.device):
            # During CUDA graph capture the current stream is a torch-owned,
            # ephemeral capture stream (true for piecewise/inductor graphs used
            # by MTP and spec-decode). Stream affinity does not apply: the
            # captured kernel replays on the caller's stream, not this one, so
            # skip the affinity guard instead of rejecting the capture stream.
            return
        self._bind_stream_key(_current_stream_key(self.device, stream))

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._closed:
            return False
        if inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        inp_bytes = inp.numel() * inp.element_size()
        if inp_bytes > self.max_size:
            return False
        if inp_bytes % 16 != 0:
            return False
        return _is_weak_contiguous(inp)

    def should_route_plain_allreduce(
        self,
        inp: torch.Tensor,
        *,
        generic_max_bytes: int,
        graph_capture: bool = False,
    ) -> bool:
        """Return whether plain all-reduce should use B12X instead of NCCL.

        ``should_allreduce`` defines functional support. This method adds the
        latency policy used by callers with an NCCL fallback. Prepared TP2 CUDA
        graph channels extend the caller's generic band through 64 storage-dense
        FP16/BF16 rows of width 4096. Every other shape and execution mode uses
        the caller-supplied size policy.
        """

        if not self.should_allreduce(inp):
            return False
        generic_max_bytes = int(generic_max_bytes)
        if generic_max_bytes < 0:
            raise ValueError("generic_max_bytes must be non-negative")
        inp_bytes = inp.numel() * inp.element_size()
        if self.world_size == 2:
            remote_push = (
                graph_capture
                and self._eager_ptrs is not None
                and self._transport_policy[2]
                and (rows := _tp2_plain_remote_push_rows(inp)) is not None
                and rows <= TP2_PLAIN_REMOTE_PUSH_AUTO_MAX_ROWS
            )
            return remote_push or inp_bytes <= generic_max_bytes
        return inp_bytes <= generic_max_bytes

    def register_buffer(self, peer_input_ptrs: Sequence[int]) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if len(peer_input_ptrs) != self.world_size:
            raise ValueError("peer_input_ptrs must match world size")
        ptrs = tuple(int(ptr) for ptr in peer_input_ptrs)
        local_ptr = ptrs[self.rank]
        existing = self._registered_input_ptrs.get(local_ptr)
        if existing is not None:
            if existing != ptrs:
                raise ValueError(
                    "input pointer is already registered with different peer_input_ptrs"
                )
            return
        self._ext.register_buffer(self._ptr, list(ptrs))
        self._registered_input_ptrs[local_ptr] = ptrs

    def _prepare_prepared_metadata(self, query) -> None:
        """Install a direct graph plan before its producer publishes pointers."""
        class _MetadataInput:
            def __init__(self, setup):
                self.shape = tuple(setup["shape"])
                self.dtype = getattr(torch, setup["dtype"])
                self.device = self_device
                self._stride = tuple(setup["stride"])

            @property
            def ndim(self):
                return len(self.shape)

            def numel(self):
                total = 1
                for extent in self.shape:
                    total *= extent
                return total

            def element_size(self):
                return self.dtype.itemsize

            def stride(self):
                return self._stride

        if _is_current_stream_capturing(self.device):
            raise RuntimeError("oneshot preparation must complete before CUDA graph capture")
        self._check_stream()
        setup = query.setup
        self_device = self.device
        state = self._ext._state(self._ptr)
        if query.surface.endswith(".all_reduce"):
            metadata = _MetadataInput(setup)
            state.plain_graph_plans[
                self._ext._plain_graph_plan_key(metadata)
            ] = self._ext._plain_graph_plan(state, metadata)

    def _prepare_prepared_surface(self, query, inp: torch.Tensor) -> None:
        """Install graph metadata during session materialization, never at run."""

        if _is_current_stream_capturing(self.device):
            raise RuntimeError("oneshot preparation must complete before CUDA graph capture")
        self._check_stream()
        setup = query.setup
        if (tuple(inp.shape), tuple(inp.stride()), _dtype_name(inp.dtype), inp.device.index) != (
            tuple(setup["shape"]), tuple(setup["stride"]), setup["dtype"], setup["device_index"],
        ):
            raise ValueError("oneshot input metadata differs from the declaration")
        state = self._ext._state(self._ptr)
        if state.eager_tables is None and int(inp.data_ptr()) not in self._registered_input_ptrs:
            raise ValueError("registered graph input is absent from the native runtime")
        if query.surface.endswith(".all_reduce"):
            state.plain_graph_plans[self._ext._plain_graph_plan_key(inp)] = (
                self._ext._plain_graph_plan(state, inp)
            )

    def _prepared_launcher(self, launchers, inp: torch.Tensor):
        state = self._ext._state(self._ptr)
        capturing = _is_current_stream_capturing(inp.device)
        _enable_device_slot_selection(state, capturing=capturing)
        key = (bool(state.device_slot_selection), int(state.slot_bias) & 1 if state.device_slot_selection else 0)
        try:
            return launchers[key], state
        except KeyError as exc:
            raise RuntimeError("oneshot slot variant was not prepared") from exc

    def _run_prepared_plain(self, inp: torch.Tensor, out: torch.Tensor, query, launchers) -> None:
        launcher, state = self._prepared_launcher(launchers, inp)
        call = query.call
        plan = state.plain_graph_plans.get(self._ext._plain_graph_plan_key(inp))
        if plan is None:
            raise RuntimeError("oneshot graph resources were not bound during preparation")
        table_address, _ = self._ext._select_table(state, inp.data_ptr())
        with torch.cuda.device(inp.device):
            launcher(
                table_address, state.signal_table_address, inp.data_ptr(), out.data_ptr(),
                call["size_packs"], call["remote_push_region_packs"],
                int(state.eager_buffer_bytes or 0) // 16,
                *plan.remote_push_slot_ptrs,
                call["blocks"] if state.device_slot_selection else call["eager_blocks"],
            )

    def _run_prepared_fused(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        out: torch.Tensor, residual_out: torch.Tensor, epsilon: float, query, launchers,
    ) -> None:
        launcher, state = self._prepared_launcher(launchers, inp)
        call = query.call
        table_address, _ = self._ext._select_table(state, inp.data_ptr())
        with torch.cuda.device(inp.device):
            launcher(
                table_address, state.signal_table_address, inp.data_ptr(), residual.data_ptr(),
                weight.data_ptr(), out.data_ptr(), residual_out.data_ptr(),
                call["hidden_packs"], call["rows"], call["ctas_per_row"],
                int(state.eager_buffer_bytes or inp.numel() * inp.element_size()) // 16,
                float(epsilon), call["blocks"],
            )


    def _prepare_input(
        self,
        inp: torch.Tensor,
        peer_input_ptrs: Optional[Sequence[int]],
    ) -> None:
        local_ptr = int(inp.data_ptr())
        if peer_input_ptrs is not None:
            if len(peer_input_ptrs) != self.world_size:
                raise ValueError("peer_input_ptrs must match world size")
            ptrs = tuple(int(ptr) for ptr in peer_input_ptrs)
            if ptrs[self.rank] != local_ptr:
                raise ValueError("peer_input_ptrs[self.rank] must match inp.data_ptr()")
            self.register_buffer(ptrs)
        elif self._eager_ptrs is None and local_ptr not in self._registered_input_ptrs:
            raise ValueError(
                "peer_input_ptrs are required unless eager IPC buffers are configured "
                "or this input was already registered"
            )

    def get_graph_buffer_ipc_meta(self) -> tuple[list[int], list[int]]:
        if self._closed:
            raise RuntimeError("runtime is closed")
        handle, offsets = self._ext.get_graph_buffer_ipc_meta(self._ptr)
        return list(handle), list(offsets)

    def register_graph_buffers_from_ranks(
        self,
        handles: Sequence[Sequence[int]],
        offsets: Sequence[Sequence[int]],
    ) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if len(handles) != self.world_size:
            raise ValueError("handles must match world size")
        if len(offsets) != self.world_size:
            raise ValueError("offsets must match world size")
        self._ext.register_graph_buffers(
            self._ptr,
            [list(map(int, handle)) for handle in handles],
            [list(map(int, rank_offsets)) for rank_offsets in offsets],
        )
    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        with _device_guard(self.device):
            return self._all_reduce_on_device(
                inp, plan=plan, out=out, peer_input_ptrs=peer_input_ptrs,
            )

    def _all_reduce_on_device(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if inp.device != self.device:
            raise ValueError(
                f"input device {inp.device} does not match runtime device {self.device}"
            )
        self._check_stream()
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy device/dtype/size/alignment/contiguity requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if peer_input_ptrs is not None:
            raise ValueError("peer input registration belongs to oneshot preparation")
        if out is None:
            out = torch.empty_like(inp)
        if out.device != inp.device:
            raise ValueError("output tensor must be on the same device as the input")
        if out.shape != inp.shape or out.dtype != inp.dtype:
            raise ValueError("output tensor must match input shape and dtype")
        if not _is_weak_contiguous(out):
            raise ValueError("output tensor must be weak-contiguous")
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        state.run_plain(inp, out)
        return out

    def all_reduce_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce ``inp``, add ``residual``, and apply RMSNorm."""
        with _device_guard(self.device):
            return self._all_reduce_fused_on_device(
                inp, residual, weight, epsilon, plan=plan, out=out,
                residual_out=residual_out, peer_input_ptrs=peer_input_ptrs,
            )
    def _all_reduce_fused_on_device(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Guarded implementation for the fused public entry point."""

        if self._closed:
            raise RuntimeError("runtime is closed")
        if inp.device != self.device:
            raise ValueError(
                f"input device {inp.device} does not match runtime device {self.device}"
            )
        self._check_stream()
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy device/dtype/size/alignment/contiguity "
                f"requirements (shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if inp.ndim == 0:
            raise ValueError("input must have at least one dimension")
        hidden_size = inp.shape[-1]
        if hidden_size * inp.element_size() % 16 != 0:
            raise ValueError(
                "the last input dimension must occupy a multiple of 16 bytes"
            )
        if residual.device != inp.device:
            raise ValueError("residual tensor must be on the same device as the input")
        if residual.shape != inp.shape or residual.dtype != inp.dtype:
            raise ValueError("residual tensor must match input shape and dtype")
        if not _is_weak_contiguous(residual):
            raise ValueError("residual tensor must be weak-contiguous")
        if weight.device != inp.device:
            raise ValueError("weight tensor must be on the same device as the input")
        if weight.shape != (hidden_size,) or weight.dtype != inp.dtype:
            raise ValueError(
                "weight tensor must match the input dtype and last dimension"
            )
        if not weight.is_contiguous():
            raise ValueError("weight tensor must be contiguous")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        if out is None:
            out = torch.empty_like(inp)
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        for name, tensor in (("output", out), ("residual output", residual_out)):
            if tensor.device != inp.device:
                raise ValueError(
                    f"{name} tensor must be on the same device as the input"
                )
            if tensor.shape != inp.shape or tensor.dtype != inp.dtype:
                raise ValueError(f"{name} tensor must match input shape and dtype")
            if not _is_weak_contiguous(tensor):
                raise ValueError(f"{name} tensor must be weak-contiguous")
        if out.data_ptr() == residual_out.data_ptr():
            raise ValueError("output and residual output must not alias")

        if peer_input_ptrs is not None:
            raise ValueError("peer input registration belongs to oneshot preparation")
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        state.run_fused(inp, residual, weight, out, residual_out, float(epsilon))
        return out, residual_out

    @contextmanager
    def capture(self, stream: object = None):
        if self.exchange_group is None and self._eager_ptrs is None:
            raise ValueError(
                "exchange_group is required for CUDA graph capture registration"
            )
        self._check_stream(stream)
        try:
            yield
        finally:
            if self.exchange_group is not None and self._eager_ptrs is None:
                self.register_graph_buffers()

    def register_graph_buffers(self) -> None:
        if self.exchange_group is None:
            raise ValueError("exchange_group is required to register graph buffers")
        local_meta = self.get_graph_buffer_ipc_meta()
        all_meta = _broadcast_gather_object(local_meta, self.exchange_group)
        num_buffers = [len(entry[1]) for entry in all_meta]
        if any(count != num_buffers[0] for count in num_buffers):
            raise RuntimeError(
                "graph capture registered a different number of buffers across ranks"
            )
        if num_buffers[0] == 0:
            return
        self.register_graph_buffers_from_ranks(
            [entry[0] for entry in all_meta],
            [entry[1] for entry in all_meta],
        )


    def _closed_import_indices(self) -> set[tuple[int, int]]:
        closed = getattr(self, "_closed_ipc_import_indices", None)
        if closed is None:
            closed = set()
            self._closed_ipc_import_indices = closed
        return closed

    def _all_python_ipc_imports_closed(self, closed: set[tuple[int, int]]) -> bool:
        if self._ipc is None:
            return not any(shared.remote_ptrs for shared in self._owned_buffers)
        return all(
            (buffer_index, remote_index) in closed
            for buffer_index, shared in enumerate(self._owned_buffers)
            for remote_index, _ in enumerate(shared.remote_ptrs)
        )

    def _close_ipc_imports_strict(self) -> None:
        if self._ipc_imports_closed:
            return
        self._closed = True
        failures: list[tuple[str, Exception]] = []
        if getattr(self, "_ptr", 0):
            try:
                self._ext.dispose(self._ptr)
            except Exception as exc:
                failures.append(("native runtime", exc))
            else:
                self._ptr = 0

        closed = self._closed_import_indices()
        if self._ipc is not None:
            for buffer_index, shared in enumerate(self._owned_buffers):
                for remote_index, ptr in enumerate(shared.remote_ptrs):
                    key = (buffer_index, remote_index)
                    if key in closed:
                        continue
                    try:
                        self._ipc.cudaIpcCloseMemHandle(ptr)
                    except Exception as exc:
                        failures.append((f"CUDA IPC import {ptr}", exc))
                    else:
                        closed.add(key)
        elif any(shared.remote_ptrs for shared in self._owned_buffers):
            failures.append(
                (
                    "CUDA IPC imports",
                    RuntimeError("CUDA runtime is unavailable for IPC unmap"),
                )
            )

        if (
            not failures
            and not getattr(self, "_ptr", 0)
            and self._all_python_ipc_imports_closed(closed)
        ):
            self._ipc_imports_closed = True
        if failures:
            _raise_local_cleanup_errors("PCIe oneshot", "IPC import close", failures)

    def _free_ipc_exports_strict(self) -> None:
        if self._ipc_exports_freed:
            return
        self._close_ipc_imports_strict()
        failures: list[tuple[str, Exception]] = []
        remaining = []
        if self._ipc is not None:
            for shared in self._owned_buffers:
                try:
                    self._ipc.cudaFree(shared.local_ptr)
                except Exception as exc:
                    remaining.append(shared)
                    failures.append((f"CUDA IPC export {shared.local_ptr}", exc))
        elif self._owned_buffers:
            remaining = list(self._owned_buffers)
            failures.append(
                (
                    "CUDA IPC exports",
                    RuntimeError("CUDA runtime is unavailable for export free"),
                )
            )
        self._owned_buffers = remaining
        if not remaining:
            self._registered_input_ptrs.clear()
            self._ipc_exports_freed = True
        if failures:
            _raise_local_cleanup_errors("PCIe oneshot", "IPC export free", failures)

    def close(self) -> None:
        if getattr(self, "_coordinated_close_complete", False):
            return
        _coordinated_close_channels(
            (self,),
            exchange_group=self.exchange_group,
            device=self.device,
        )

    def __del__(
        self,
        _quarantine: dict[int, object] = _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    ) -> None:
        # GC cannot prove that work queued on an arbitrary CUDA stream has
        # completed, and it must not synchronize or enter a distributed
        # barrier.  Retain the whole runtime: rank_data is a torch allocation
        # whose raw pointer is stored in the native object and captured graph
        # arguments, so retaining only IPC mappings/exports is insufficient.
        if getattr(self, "_coordinated_close_complete", False):
            return
        if (
            getattr(self, "_ptr", 0)
            or hasattr(self, "rank_data")
            or getattr(self, "_owned_buffers", ())
        ):
            _quarantine[id(self)] = self


def _is_current_stream_capturing(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    return bool(is_capturing())


class PCIeOneshotAllReducePool:
    """Stream-affine PCIe oneshot wrapper.

    A ``PCIeOneshotAllReduce`` instance is a single ordered channel with one
    signal buffer and one double-buffered staging pair. The pool creates a
    separate channel for each CUDA stream key so multi-stream callers never
    reuse those buffers concurrently.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        ipc: Optional[CudaRTLibrary] = None,
        single_channel: bool = False,
        max_concurrent_channels: int = 1,
        channel_factory: Optional[
            Callable[[Optional[int]], PCIeOneshotAllReduce]
        ] = None,
    ):
        resolved_group = _resolve_exchange_group(exchange_group, process_group)

        def normalize_and_validate():
            normalized_rank = int(rank)
            normalized_world_size = int(world_size)
            device_obj = _normalize_device(device)
            normalized_eager_bytes = int(eager_buffer_bytes)
            normalized_max_size = int(max_size)
            normalized_rank_data_bytes = int(rank_data_bytes)
            normalized_single_channel = bool(single_channel)
            normalized_max_concurrent_channels = int(max_concurrent_channels)
            normalized_transport_policy = _transport_policy_contract()
            if normalized_world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {normalized_world_size}")
            if not 0 <= normalized_rank < normalized_world_size:
                raise ValueError(
                    f"invalid rank {normalized_rank} for world size "
                    f"{normalized_world_size}"
                )
            if normalized_eager_bytes <= 0:
                raise ValueError("eager_buffer_bytes must be positive")
            if normalized_max_size <= 0:
                raise ValueError("max_size must be positive")
            if normalized_max_size > normalized_eager_bytes:
                raise ValueError("max_size exceeds eager buffer capacity")
            if normalized_rank_data_bytes <= 0:
                raise ValueError("rank_data_bytes must be positive")
            if normalized_max_concurrent_channels <= 0:
                raise ValueError("max_concurrent_channels must be positive")
            if channel_factory is None:
                if resolved_group is None:
                    raise ValueError(
                        "exchange_group is required unless channel_factory is provided"
                    )
                if device_obj.type != "cuda":
                    raise ValueError("PCIe oneshot pool requires a CUDA device")
                group_rank = dist.get_rank(group=resolved_group)
                group_world_size = dist.get_world_size(group=resolved_group)
                if normalized_rank != group_rank:
                    raise ValueError(
                        f"supplied rank {normalized_rank} does not match process "
                        f"group rank {group_rank}"
                    )
                if normalized_world_size != group_world_size:
                    raise ValueError(
                        f"supplied world size {normalized_world_size} does not "
                        f"match process group size {group_world_size}"
                    )
            return (
                normalized_rank,
                normalized_world_size,
                device_obj,
                normalized_eager_bytes,
                normalized_max_size,
                normalized_rank_data_bytes,
                normalized_single_channel,
                normalized_max_concurrent_channels,
                normalized_transport_policy,
            )

        if channel_factory is None and resolved_group is not None:
            normalized = _run_collective_preallocation_setup(
                owner="PCIe oneshot pool argument validation",
                exchange_group=resolved_group,
                setup=normalize_and_validate,
            )
        else:
            normalized = normalize_and_validate()

        (
            self.rank,
            self.world_size,
            self.device,
            self.eager_buffer_bytes,
            self.max_size,
            self.rank_data_bytes,
            self.single_channel,
            self.max_concurrent_channels,
            self._transport_policy,
        ) = normalized
        self._sharded_eager_storage = _uses_sharded_eager_storage(
            self.world_size,
            self._transport_policy,
        )
        self._eager_storage_shards = _eager_storage_shards(
            self.world_size,
            self._transport_policy,
        )
        self.exchange_group = resolved_group
        self.process_group = self.exchange_group
        self._channel_factory = channel_factory
        self._channels: dict[int, PCIeOneshotAllReduce] = {}
        self._logical_channels: dict[str, PCIeOneshotAllReduce] = {}
        self._captured_channel_ids: set[str] = set()
        self._all_channels: list[PCIeOneshotAllReduce] = []
        self._capture_channel_stack: list[PCIeOneshotAllReduce] = []
        self._closed = False

        self._ipc = ipc
        self._ext = ext_module
        self._signal_bytes = 0
        if self._channel_factory is None:
            assert self.exchange_group is not None
            _require_full_grid_residency(
                owner="PCIe oneshot",
                required_sms=(ONESHOT_REQUIRED_SMS * self.max_concurrent_channels),
                device=self.device,
                exchange_group=self.exchange_group,
            )

            def prepare() -> tuple[CudaRTLibrary, object, int]:
                prepared_ipc = self._ipc or CudaRTLibrary()
                prepared_ipc.cudaSetDevice(_cuda_device_index(self.device))
                prepared_ext = self._ext or _load_extension()
                return (
                    prepared_ipc,
                    prepared_ext,
                    int(prepared_ext.meta_size()),
                )

            self._ipc, self._ext, self._signal_bytes = (
                _run_collective_preallocation_setup(
                    owner="PCIe oneshot pool",
                    exchange_group=self.exchange_group,
                    setup=prepare,
                )
            )
            _require_collective_contract(
                owner="PCIe oneshot pool channel layout",
                exchange_group=self.exchange_group,
                contract=(
                    self._signal_bytes,
                    self.eager_buffer_bytes,
                    self.max_size,
                    self.rank_data_bytes,
                    self.single_channel,
                    self.max_concurrent_channels,
                    self._transport_policy,
                ),
            )
            # A genuine single-channel pool has no independent eager/graph
            # owners to name. Multi-channel distributed callers must prepare
            # explicit semantic ids before their first eager operation.
            if self.single_channel:
                self.prepare_channels((_SINGLE_CHANNEL_ID,))

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
        max_concurrent_channels: int = 1,
    ) -> "PCIeOneshotAllReducePool":
        return cls(
            rank=dist.get_rank(group=exchange_group),
            world_size=dist.get_world_size(group=exchange_group),
            device=device,
            exchange_group=exchange_group,
            eager_buffer_bytes=eager_buffer_bytes,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            single_channel=single_channel,
            max_concurrent_channels=max_concurrent_channels,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: Optional[int] = None,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
        max_concurrent_channels: int = 1,
    ) -> "PCIeOneshotAllReducePool":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=max_input_bytes
            if eager_buffer_bytes is None
            else eager_buffer_bytes,
            max_size=max_input_bytes if max_size is None else max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            single_channel=single_channel,
            max_concurrent_channels=max_concurrent_channels,
        )

    def _new_channel(self, stream_key: Optional[int]) -> PCIeOneshotAllReduce:
        if self._channel_factory is not None:
            channel = self._channel_factory(stream_key)
            if self.single_channel:
                channel._stream_affine = False
            channel._bind_stream_key(stream_key)
            self._all_channels.append(channel)
            return channel

        if self.exchange_group is None or self._ipc is None or self._ext is None:
            raise RuntimeError("pool is not configured to allocate channels")

        channel_buffers = PCIeOneshotAllReduce._allocate_eager_channel_buffers(
            self.exchange_group,
            signal_bytes=self._signal_bytes,
            eager_buffer_bytes=self.eager_buffer_bytes,
            eager_storage_shards=self._eager_storage_shards,
            ipc=self._ipc,
        )
        owned_buffers = [channel_buffers.owned_buffer]
        channel, init_error = PCIeOneshotAllReduce._from_prepared_factory(
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            signal_ptrs=channel_buffers.signal_ptrs,
            eager_buffer_ptrs0=channel_buffers.eager0_ptrs,
            eager_buffer_ptrs1=channel_buffers.eager1_ptrs,
            eager_buffer_bytes=self.eager_buffer_bytes,
            transport_policy=self._transport_policy,
            exchange_group=self.exchange_group,
            ipc=self._ipc,
            owned_buffers=owned_buffers,
            max_size=self.max_size,
            rank_data_bytes=self.rank_data_bytes,
            ext_module=self._ext,
            stream_affine=not self.single_channel,
        )

        def abort_native_runtime() -> None:
            pointer = getattr(channel, "_ptr", 0)
            if pointer:
                self._ext.dispose(pointer)
                channel._ptr = 0
            if hasattr(channel, "rank_data"):
                del channel.rank_data

        def detach_shared_ownership() -> None:
            channel._owned_buffers.clear()

        _finish_collective_runtime_setup(
            owner="PCIe oneshot pool channel",
            exchange_group=self.exchange_group,
            ipc=self._ipc,
            shared=channel_buffers.owned_buffer,
            local_error=init_error,
            local_cleanup=abort_native_runtime,
            detach_shared_ownership=detach_shared_ownership,
        )
        channel._bind_stream_key(stream_key)
        self._all_channels.append(channel)
        return channel

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        """Collectively allocate named channels in canonical order.

        Local CUDA stream handles are process-local and cannot identify a
        distributed channel. Every rank may supply the same set in any order;
        the sorted logical ids define identical IPC allocation order.
        """

        if self._channel_factory is None:
            assert self.exchange_group is not None

            def normalize_and_validate() -> tuple[str, ...]:
                if self._closed:
                    raise RuntimeError("pool is closed")
                return tuple(
                    sorted(
                        {_normalize_logical_channel_id(value) for value in channel_ids}
                    )
                )

            # Normalize and validate through a status collective first.  A
            # rank-local empty/invalid id must not return or raise while peers
            # enter the subsequent contract exchange or IPC allocation.
            normalized = _run_collective_preallocation_setup(
                owner="PCIe oneshot logical channel validation",
                exchange_group=self.exchange_group,
                setup=normalize_and_validate,
            )
            local_state = (normalized, tuple(sorted(self._logical_channels)))
            gathered = _broadcast_gather_object(local_state, self.exchange_group)
            if any(state != local_state for state in gathered):
                raise RuntimeError(
                    "PCIe oneshot logical channel preparation differs across "
                    f"ranks: {gathered}"
                )
        else:
            if self._closed:
                raise RuntimeError("pool is closed")
            normalized = tuple(
                sorted({_normalize_logical_channel_id(value) for value in channel_ids})
            )

        if not normalized:
            return

        for channel_id in normalized:
            if channel_id in self._logical_channels:
                continue
            self._logical_channels[channel_id] = self._new_channel(None)

    def checkpoint_channels(
        self,
    ) -> tuple[
        int,
        dict[int, PCIeOneshotAllReduce],
        dict[str, PCIeOneshotAllReduce],
        set[str],
    ]:
        """Snapshot channel ownership before a throwaway graph capture."""
        if self._closed:
            raise RuntimeError("pool is closed")
        if self._capture_channel_stack:
            raise RuntimeError("cannot checkpoint channels during capture")
        return (
            len(self._all_channels),
            dict(self._channels),
            dict(self._logical_channels),
            set(self._captured_channel_ids),
        )

    def rollback_channels(
        self,
        checkpoint: tuple,
    ) -> None:
        """Close channels created after ``checkpoint`` and restore mappings.

        Callers must destroy and synchronize any graphs that reference the
        transient channels before rolling back.
        """
        if self._closed:
            raise RuntimeError("pool is closed")
        if self._capture_channel_stack:
            raise RuntimeError("cannot roll back channels during capture")
        if len(checkpoint) == 2:
            all_channels_len, channels = checkpoint
            logical_channels = dict(self._logical_channels)
            captured_channel_ids = set(self._captured_channel_ids)
        elif len(checkpoint) == 4:
            (
                all_channels_len,
                channels,
                logical_channels,
                captured_channel_ids,
            ) = checkpoint
        else:
            raise ValueError("invalid channel checkpoint")
        if not 0 <= all_channels_len <= len(self._all_channels):
            raise ValueError("channel checkpoint does not belong to this pool")

        retained = self._all_channels[:all_channels_len]
        retained_ids = {id(channel) for channel in retained}
        transient = self._all_channels[all_channels_len:]

        channels_to_close = tuple(
            dict.fromkeys(
                channel for channel in transient if id(channel) not in retained_ids
            )
        )
        _coordinated_close_channels(
            channels_to_close,
            exchange_group=self.exchange_group,
            device=self.device,
        )
        self._all_channels = retained
        self._channels = dict(channels)
        self._logical_channels = dict(logical_channels)
        self._captured_channel_ids = set(captured_channel_ids)

    def for_stream(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ) -> PCIeOneshotAllReduce:
        if self._closed:
            raise RuntimeError("pool is closed")
        capturing = _is_current_stream_capturing(self.device)
        if capturing:
            if not self._capture_channel_stack:
                raise RuntimeError(
                    "PCIe oneshot CUDA graph capture requires an active "
                    "pool.capture() graph-owned channel context"
                )
            channel = self._capture_channel_stack[-1]
            if not self.single_channel:
                stream_key = _current_stream_key(self.device, stream)
                channel_key = 0 if stream_key is None else int(stream_key)
                self._channels[channel_key] = channel
            return channel
        if self.single_channel:
            channel = self._channels.get(0)
            if channel is not None:
                return channel
            if self._channel_factory is None:
                channel = self._logical_channels[_SINGLE_CHANNEL_ID]
                self._channels[0] = channel
                return channel
            if _is_current_stream_capturing(self.device):
                raise RuntimeError(
                    "PCIe oneshot pool has no channel to reuse during CUDA graph "
                    "capture; perform an eager all-reduce (or call for_stream) "
                    "before capture starts"
                )
            channel = self._new_channel(None)
            self._channels[0] = channel
            return channel

        stream_key = _current_stream_key(self.device, stream)
        channel_key = 0 if stream_key is None else int(stream_key)
        if self._capture_channel_stack:
            # The semantic capture scope starts before vLLM's eager graph
            # warmup. Route that warmup through the graph-owned channel too,
            # even though CUDA capture has not started yet. Once CUDA capture
            # begins, torch/Inductor may replace the enclosing stream with an
            # ephemeral nested capture stream; that key is deliberately not
            # made the channel's permanent owner because graph replay runs on
            # the enclosing stream.
            channel = self._capture_channel_stack[-1]
            if not _is_current_stream_capturing(self.device):
                channel._bind_stream_key(stream_key)
            self._channels[channel_key] = channel
            return channel

        if self._channel_factory is None:
            if channel_id is None:
                raise RuntimeError(
                    "distributed PCIe oneshot eager use requires an explicit "
                    "semantic channel_id shared by every rank"
                )
            logical_id = _normalize_logical_channel_id(channel_id)
            channel = self._logical_channels.get(logical_id)
            if channel is None:
                raise RuntimeError(
                    f"logical channel {logical_id!r} is not prepared; call "
                    "prepare_channels() collectively before use"
                )
            mapped = self._channels.get(channel_key)
            if mapped is not None and mapped is not channel:
                raise RuntimeError(
                    f"CUDA stream key {channel_key} is already bound to another "
                    "logical PCIe oneshot channel"
                )
            channel._bind_stream_key(stream_key)
            self._channels[channel_key] = channel
            return channel
        channel = self._channels.get(channel_key)
        if channel is not None:
            return channel
        channel = self._new_channel(stream_key)
        self._channels[channel_key] = channel
        return channel
    def _prepared_channel_for_stream(
        self, stream: object, channel_id: Optional[str],
    ) -> PCIeOneshotAllReduce:
        """Select an extant channel only; replay must never allocate one."""
        if self._capture_channel_stack:
            return self._capture_channel_stack[-1]
        if self.single_channel:
            channel = self._channels.get(0)
            if channel is None and self._channel_factory is None:
                channel = self._logical_channels.get(_SINGLE_CHANNEL_ID)
            if channel is None:
                raise RuntimeError("PCIe oneshot channel was not prepared")
            return channel
        stream_key = _current_stream_key(self.device, stream)
        channel_key = 0 if stream_key is None else int(stream_key)
        channel = self._channels.get(channel_key)
        if channel is not None:
            return channel
        if self._channel_factory is not None:
            raise RuntimeError("PCIe oneshot stream channel was not prepared")
        if channel_id is None:
            raise RuntimeError("prepared PCIe oneshot use requires its semantic channel_id")
        channel = self._logical_channels.get(_normalize_logical_channel_id(channel_id))
        if channel is None:
            raise RuntimeError("PCIe oneshot logical channel was not prepared")
        channel._bind_stream_key(stream_key)
        self._channels[channel_key] = channel
        return channel

    def all_reduce(
        self, inp: torch.Tensor, *, plan: Plan, out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None, stream: object = None,
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        with _device_guard(self.device):
            return self._all_reduce_on_device(
                inp, plan=plan, out=out, peer_input_ptrs=peer_input_ptrs,
                stream=stream, channel_id=channel_id,
            )

    def _all_reduce_on_device(
        self, inp: torch.Tensor, *, plan: Plan, out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None, stream: object = None,
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        state = require_prepared(plan, "comm.pcie", self.device)
        channel = self._prepared_channel_for_stream(stream, channel_id)
        state.require_runtime(channel)
        if stream is not None and self.device.type == "cuda":
            with torch.cuda.stream(stream):
                return channel.all_reduce(
                    inp, plan=plan, out=out, peer_input_ptrs=peer_input_ptrs
                )
        return channel.all_reduce(inp, plan=plan, out=out, peer_input_ptrs=peer_input_ptrs)

    def all_reduce_fused_add_rms_norm(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float,
        *, plan: Plan, out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None, stream: object = None,
        channel_id: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _device_guard(self.device):
            return self._all_reduce_fused_on_device(
                inp, residual, weight, epsilon, plan=plan, out=out,
                residual_out=residual_out, peer_input_ptrs=peer_input_ptrs,
                stream=stream, channel_id=channel_id,
            )

    def _all_reduce_fused_on_device(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float,
        *, plan: Plan, out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None, stream: object = None,
        channel_id: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = require_prepared(plan, "comm.pcie", self.device)
        channel = self._prepared_channel_for_stream(stream, channel_id)
        state.require_runtime(channel)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            return channel.all_reduce_fused_add_rms_norm(
                inp, residual, weight, epsilon, plan=plan, out=out,
                residual_out=residual_out, peer_input_ptrs=peer_input_ptrs,
            )

        if stream is not None and self.device.type == "cuda":
            with torch.cuda.stream(stream):
                return run()
        return run()

    @contextmanager
    def capture(self, stream: object = None, *, channel_id: Optional[str] = None):
        """Capture on a globally named channel.

        An explicit id names an independently replayable graph. First use of
        an unprepared explicit id is a collective convenience path: every rank
        must enter with the same id or all ranks fail closed. Production
        callers that know the full graph set should call prepare_channels()
        once before capture; after that collective preparation, ranks may
        capture members of the agreed catalog in different orders. Each graph
        must still replay with its same-id peers using collectively compatible
        kernel sequences and shapes.

        Distributed pools require this semantic id. A local capture ordinal or
        CUDA stream handle cannot distinguish target and draft graphs when
        ranks construct them in a different order, so guessing is unsafe.
        """
        previous_channels: Optional[dict[int, PCIeOneshotAllReduce]] = None
        if not self.single_channel and _is_current_stream_capturing(self.device):
            raise RuntimeError(
                "PCIe oneshot capture context must be entered before CUDA graph "
                "capture starts"
            )
        if self.single_channel:
            channel = self.for_stream(
                stream,
                channel_id=_SINGLE_CHANNEL_ID if channel_id is None else channel_id,
            )
        elif self._channel_factory is None:
            assert self.exchange_group is not None

            def validate_capture_id() -> str:
                if channel_id is None:
                    raise RuntimeError(
                        "distributed PCIe oneshot capture requires a stable "
                        "semantic channel_id shared by every rank"
                    )
                logical_id = _normalize_logical_channel_id(channel_id)
                if logical_id in self._captured_channel_ids:
                    raise RuntimeError(
                        f"logical channel {logical_id!r} was already captured; "
                        "each independently replayable graph requires a unique id"
                    )
                return logical_id

            logical_id = _run_collective_preallocation_setup(
                owner="PCIe oneshot capture channel validation",
                exchange_group=self.exchange_group,
                setup=validate_capture_id,
            )
            needs_preparation = _collective_capture_needs_preparation(
                owner="PCIe oneshot",
                logical_id=logical_id,
                prepared_channel_ids=self._logical_channels,
                exchange_group=self.exchange_group,
            )
            if needs_preparation:
                self.prepare_channels((logical_id,))
            previous_channels = dict(self._channels)
            stream_key = _current_stream_key(self.device, stream)
            channel_key = 0 if stream_key is None else int(stream_key)
            channel = self._logical_channels[logical_id]
            channel._bind_stream_key(stream_key)
            self._channels[channel_key] = channel
            self._captured_channel_ids.add(logical_id)
        else:
            stream_key = _current_stream_key(self.device, stream)
            channel_key = 0 if stream_key is None else int(stream_key)
            # A CUDA stream handle can be recycled after one graph manager
            # finishes capture. Graphs retain the channel pointers, so a new
            # manager must receive a fresh channel even when the numeric stream
            # key is identical. _all_channels keeps replaced channels alive.
            previous_channels = dict(self._channels)
            channel = self._new_channel(stream_key)
            self._channels[channel_key] = channel
        with channel.capture(stream=stream):
            self._capture_channel_stack.append(channel)
            try:
                yield channel
            finally:
                popped = self._capture_channel_stack.pop()
                if popped is not channel:
                    raise RuntimeError("PCIe oneshot capture channel stack corrupted")
                if previous_channels is not None:
                    # Graph nodes retain this channel through _all_channels.
                    # Restore every alias in strict LIFO order so nested or
                    # recycled torch capture-stream keys cannot escape into a
                    # later independent/unwrapped capture.
                    for key, mapped in tuple(self._channels.items()):
                        if mapped is not channel:
                            continue
                        previous = previous_channels.get(key)
                        if previous is None:
                            del self._channels[key]
                        else:
                            self._channels[key] = previous

    def close(self) -> None:
        if self._closed:
            return
        seen: set[int] = set()
        channels = []
        for channel in (*self._all_channels, *self._channels.values()):
            if id(channel) not in seen:
                seen.add(id(channel))
                channels.append(channel)
        _coordinated_close_channels(
            channels,
            exchange_group=self.exchange_group,
            device=self.device,
        )
        self._closed = True
        self._all_channels.clear()
        self._channels.clear()
        self._logical_channels.clear()
        self._captured_channel_ids.clear()

    def __del__(
        self,
        _quarantine: dict[int, object] = _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    ) -> None:
        # A pool may be collected while graph or stream work still references
        # its channels.  Keep the whole pool (and therefore every channel and
        # rank_data tensor) alive; never unmap, synchronize, or communicate
        # from GC. Explicit close() clears ownership before finalization.
        if not getattr(self, "_closed", True):
            _quarantine[id(self)] = self


__all__ = [
    "PCIeOneshotAllReduce",
    "PCIeOneshotAllReducePool",
    "SUPPORTED_WORLD_SIZES",
    "parse_pcie_oneshot_max_size",
]
