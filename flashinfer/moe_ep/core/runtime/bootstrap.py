"""Process-wide runtime bootstrap for moe_ep (torch.distributed, NVSHMEM, …)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import FrozenSet

from ...config import BootstrapConfig

Requirement = str
TORCH_DIST = "torch_dist"
NVSHMEM = "nvshmem"

_logger = logging.getLogger(__name__)


@dataclass
class MoEEpRuntimeHandle:
    """Opaque token returned by :func:`bootstrap_moe_ep_runtime`."""

    requirements: FrozenSet[str]


@dataclass
class _RuntimeState:
    ref_count: int = 0
    owned_torch_dist: bool = False
    owned_nvshmem: bool = False
    active_requirements: FrozenSet[str] = frozenset()


_STATE = _RuntimeState()


def _mega_no_dist() -> bool:
    return bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def _nvshmem_initialized() -> bool:
    try:
        from nvshmem.core.memory import InternalInitStatus, _is_initialized

        return _is_initialized["status"] == InternalInitStatus.INITIALIZED
    except (ImportError, AttributeError, KeyError):
        return False


def _launched_via_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _ensure_cuda_device(bootstrap: BootstrapConfig) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", str(bootstrap.rank)))
    torch.cuda.set_device(local_rank)


def ensure_moe_ep_cuda_device(bootstrap: BootstrapConfig) -> None:
    """Bind the current process to ``LOCAL_RANK`` before any CUDA allocations."""
    _ensure_cuda_device(bootstrap)


def _ensure_torch_dist(bootstrap: BootstrapConfig) -> bool:
    """Ensure ``torch.distributed`` is initialized. Returns True if this call did it."""
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        from ...core.validation.common import validate_bootstrap_world_size

        validate_bootstrap_world_size(bootstrap)
        return False

    _ensure_cuda_device(bootstrap)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", str(bootstrap.rank)))
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", device_id=device)
    elif bootstrap.world_size == 1:
        dist.init_process_group(
            backend="gloo",
            rank=bootstrap.rank,
            world_size=bootstrap.world_size,
            init_method="tcp://127.0.0.1:29500",
        )
    else:
        raise RuntimeError(
            "MoEEpLayer requires torch.distributed to be initialized before "
            "construction when RANK/WORLD_SIZE are not set in the environment."
        )

    from ...core.validation.common import validate_bootstrap_world_size

    validate_bootstrap_world_size(bootstrap)
    return True


def _init_nvshmem_after_dist(bootstrap: BootstrapConfig) -> bool:
    """Initialize NVSHMEM after ``torch.distributed`` is already up.

    Mirrors ``src.bootstrap.init_dist_and_nvshmem`` without re-initing the
    process group (avoids Gloo mesh setup on clusters where only NCCL works).
    Returns True if this call initialized NVSHMEM.
    """
    if _mega_no_dist() or _nvshmem_initialized():
        return False

    from ...backends.mega.kernel.cutedsl_backend_kernels import bootstrap_paths

    bootstrap_paths()
    import numpy as np
    import nvshmem.core
    import torch
    import torch.distributed as dist

    try:
        from cuda.core.experimental import Device
    except ImportError:
        from cuda.core import Device

    if not dist.is_initialized():
        raise RuntimeError(
            "NVSHMEM bootstrap requires torch.distributed to be initialized first."
        )

    from ..bootstrap_utils import bootstrap_comm_group, bootstrap_ep_rank_world

    pg = bootstrap_comm_group(bootstrap)
    rank, world_size = bootstrap_ep_rank_world(bootstrap)

    local_rank = int(os.environ.get("LOCAL_RANK", str(bootstrap.rank)))
    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()

    uid = nvshmem.core.get_unique_id(empty=(rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0, group=pg)
    dist.barrier(group=pg)
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)

    nvshmem.core.init(
        device=dev,
        uid=uid,
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )
    return True


def _ensure_nvshmem(bootstrap: BootstrapConfig) -> tuple[bool, bool]:
    """Ensure NVSHMEM (+ torch.distributed when needed) is initialized.

    Returns ``(owned_nvshmem, owned_torch_dist)``.
    """
    if _mega_no_dist():
        return False, False
    if _nvshmem_initialized():
        _ensure_torch_dist(bootstrap)
        return False, False

    owned_torch_dist = _ensure_torch_dist(bootstrap)
    owned_nvshmem = _init_nvshmem_after_dist(bootstrap)
    from ...core.validation.common import validate_bootstrap_world_size

    validate_bootstrap_world_size(bootstrap)
    return owned_nvshmem, owned_torch_dist


def bootstrap_moe_ep_runtime(
    bootstrap: BootstrapConfig,
    requirements: FrozenSet[str],
) -> MoEEpRuntimeHandle:
    """Acquire shared moe_ep runtime resources required by a backend."""
    if not requirements:
        return MoEEpRuntimeHandle(requirements=frozenset())

    _STATE.ref_count += 1
    _STATE.active_requirements |= requirements

    if NVSHMEM in requirements:
        owned_nvshmem, owned_torch_dist = _ensure_nvshmem(bootstrap)
        _STATE.owned_nvshmem = _STATE.owned_nvshmem or owned_nvshmem
        _STATE.owned_torch_dist = _STATE.owned_torch_dist or owned_torch_dist
    elif TORCH_DIST in requirements:
        _STATE.owned_torch_dist = _STATE.owned_torch_dist or _ensure_torch_dist(
            bootstrap
        )

    return MoEEpRuntimeHandle(requirements=frozenset(requirements))


def finalize_moe_ep_runtime(handle: MoEEpRuntimeHandle | None) -> None:
    """Release runtime resources acquired via :func:`bootstrap_moe_ep_runtime`."""
    if handle is None or not handle.requirements:
        return
    if _STATE.ref_count <= 0:
        return

    _STATE.ref_count -= 1
    if _STATE.ref_count > 0:
        return

    if _STATE.owned_nvshmem:
        try:
            import nvshmem.core

            nvshmem.core.finalize()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("moe_ep NVSHMEM finalize failed: %s", exc, exc_info=True)
        _STATE.owned_nvshmem = False

    if _STATE.owned_torch_dist and not _launched_via_torchrun():
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "moe_ep torch.distributed teardown failed: %s", exc, exc_info=True
            )
        _STATE.owned_torch_dist = False
    elif _STATE.owned_torch_dist and _launched_via_torchrun():
        # torchrun / pytest-xdist owns the process group for the whole job.
        _STATE.owned_torch_dist = False

    _STATE.active_requirements = frozenset()


def split_comm_runtime_requirements(comm_backend_name: str) -> FrozenSet[str]:
    """Runtime needs for a split-path comm backend."""
    if comm_backend_name in ("nccl_ep", "nixl_ep"):
        return frozenset({TORCH_DIST})
    return frozenset()


def nvfp4_cutedsl_runtime_requirements(bootstrap: BootstrapConfig) -> FrozenSet[str]:
    """Runtime needs for the CuTeDSL NVFP4 mega kernel."""
    if _mega_no_dist():
        return frozenset()
    return frozenset({TORCH_DIST, NVSHMEM})


def mxfp8_cutedsl_runtime_requirements(bootstrap: BootstrapConfig) -> FrozenSet[str]:
    """Runtime needs for the CuTeDSL MXFP8 mega kernel."""
    return nvfp4_cutedsl_runtime_requirements(bootstrap)


__all__ = [
    "MoEEpRuntimeHandle",
    "NVSHMEM",
    "TORCH_DIST",
    "bootstrap_moe_ep_runtime",
    "ensure_moe_ep_cuda_device",
    "finalize_moe_ep_runtime",
    "mxfp8_cutedsl_runtime_requirements",
    "nvfp4_cutedsl_runtime_requirements",
    "split_comm_runtime_requirements",
]
