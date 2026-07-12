# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared MegaMoE frontend utilities (dist bootstrap, sym heap, compile state)."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Optional, Tuple

import torch


def _no_dist() -> bool:
    # Read at call time, not import time: callers (e.g. single-rank pytest
    # tests) set MEGA_NO_DIST=1 after this module is already imported.
    return bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def bootstrap_dist():
    """Initialize torch.distributed + NVSHMEM (or single-rank CUDA when ``MEGA_NO_DIST=1``).

    Returns ``(local_rank, rank, world_size, cuda.core.Device)``.
    """
    if _no_dist():
        torch.cuda.set_device(0)
        try:
            from cuda.core.experimental import Device
        except ImportError:
            from cuda.core import Device
        dev = Device(0)
        dev.set_current()
        return 0, 0, 1, dev

    from src.bootstrap import init_dist_and_nvshmem

    return init_dist_and_nvshmem()


def sym_zeros(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Zero-initialised symmetric-heap tensor (plain CUDA when ``MEGA_NO_DIST=1``)."""
    if _no_dist():
        tensor = torch.zeros(shape, dtype=dtype, device="cuda")
        # Tag so free_sym_tensor frees by allocation kind, not by whatever
        # MEGA_NO_DIST happens to be at free time (the env can be flipped
        # back between alloc and free, e.g. by pytest monkeypatch teardown).
        tensor._mega_plain_alloc = True
        return tensor
    import nvshmem.core

    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    tensor.zero_()
    return tensor


def free_sym_tensor(tensor: Optional[torch.Tensor]) -> None:
    """Release an NVSHMEM symmetric tensor; no-op under ``MEGA_NO_DIST=1``."""
    if tensor is None or getattr(tensor, "_mega_plain_alloc", False) or _no_dist():
        return
    import nvshmem.core

    try:
        nvshmem.core.free_tensor(tensor)
    except (RuntimeError, ValueError, TypeError) as exc:
        msg = str(exc).lower()
        if any(token in msg for token in ("already", "freed", "invalid")):
            return
        raise


def _compute_peer_offsets(
    sym_tensor: torch.Tensor,
    world_size: int,
) -> Tuple[int, Tuple[int, ...]]:
    if _no_dist():
        local_base = int(sym_tensor.data_ptr())
        return local_base, tuple(0 for _ in range(world_size))
    import nvshmem.core

    local_base = int(sym_tensor.data_ptr())
    peer_offsets_list = tuple(
        int(nvshmem.core.get_peer_tensor(sym_tensor, peer).data_ptr()) - local_base
        for peer in range(world_size)
    )
    return local_base, peer_offsets_list


@dataclasses.dataclass
class _CompiledMega:
    compiled: Optional[Any]
    kernel: Any
    local_workspace: torch.Tensor
    shared_workspace: torch.Tensor
    symmetric_base: int
    peer_offsets_list: Tuple[int, ...]


def _zero_local_workspace_preserving_phase(mega: _CompiledMega) -> None:
    kernel = mega.kernel
    name = "nvlink_barrier_counter"
    if name not in kernel._local_offsets:
        mega.local_workspace.zero_()
        return

    off = int(kernel._local_offsets[name])
    nbytes = int(kernel._local_region_by_name[name].nbytes)
    total = mega.local_workspace.numel()
    if off > 0:
        mega.local_workspace[:off].zero_()
    end = off + nbytes
    if end < total:
        mega.local_workspace[end:].zero_()


def reset_compiled_mega_workspaces(mega: _CompiledMega) -> None:
    """Reset kernel workspaces before a launch (preserves NVLink barrier phase)."""
    kernel = mega.kernel
    if getattr(kernel, "world_size", 1) > 1:
        _zero_local_workspace_preserving_phase(mega)
    else:
        mega.shared_workspace.zero_()
        mega.local_workspace.zero_()
