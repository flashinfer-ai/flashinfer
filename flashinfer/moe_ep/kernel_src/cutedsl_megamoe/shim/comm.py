# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared MegaMoE frontend utilities (dist bootstrap, sym heap, compile state)."""

from __future__ import annotations

import dataclasses
import os
import warnings
from typing import Any, Optional, Tuple

import torch


def resolve_gate_up_clamp(
    *,
    gate_up_clamp: Optional[float],
    activation_clamp: Optional[float],
) -> Optional[float]:
    """Return the effective gate-up clamp, rejecting conflicting alias args."""
    if gate_up_clamp is not None and activation_clamp is not None:
        if gate_up_clamp != activation_clamp:
            raise ValueError(
                "gate_up_clamp and activation_clamp disagree "
                f"({gate_up_clamp} vs {activation_clamp}); pass only one."
            )
        warnings.warn(
            "activation_clamp is deprecated; use gate_up_clamp.",
            DeprecationWarning,
            stacklevel=3,
        )
    if gate_up_clamp is not None:
        return gate_up_clamp
    if activation_clamp is not None:
        warnings.warn(
            "activation_clamp is deprecated; use gate_up_clamp.",
            DeprecationWarning,
            stacklevel=3,
        )
        return activation_clamp
    return None


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
    from nvshmem.core.interop.torch import tensor_get_buffer

    local_base = int(sym_tensor.data_ptr())
    my_pe = int(nvshmem.core.my_pe())
    buf, _size, _dtype = tensor_get_buffer(sym_tensor)

    def _peer_base(peer: int) -> int:
        # Own rank maps to the local base (nvshmem_ptr identity).  Skipping
        # the get_peer_buffer call for it matters: nvshmem4py resolves the
        # self-peer to the PARENT tracker entry and bumps its ref count, which
        # defers the real nvshmem free from free_tensor() to GC (the "memory
        # was not freed explicitly" finalize warnings).
        if peer == my_pe:
            return local_base
        # Deliberately NOT nvshmem.core.get_peer_tensor(): its
        # ``.view(tensor.shape)`` breaks when the nvshmem heap reuses an
        # address for a smaller allocation while the nvshmem4py tracker still
        # holds a stale larger peer entry (first hit when in_kernel_fc2_reduce
        # shrank shared_workspace from ~1 MiB to ~8 KiB at a reused address).
        # Only the peer BASE ADDRESS is needed here, and the nvshmem_ptr
        # address mapping is deterministic, so read it off the peer Buffer.
        peer_buf = nvshmem.core.get_peer_buffer(buf, peer)
        return int(torch.utils.dlpack.from_dlpack(peer_buf).data_ptr())

    peer_offsets_list = tuple(
        _peer_base(peer) - local_base for peer in range(world_size)
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
    # Launch-kwargs cache: rebuilding the cute tensor views (12x from_dlpack +
    # SymBufferHost) and re-validating inputs costs real host time per launch,
    # and the launch inputs are stable session buffers in steady state.  Keyed
    # on the input data_ptrs + token count + stream; a hit skips validation
    # entirely (the same tensors were validated when the entry was built).
    # Lives here so a recompile naturally drops it.
    launch_key: Optional[tuple] = None
    launch_kwargs: Optional[dict] = None
    launch_output: Optional[torch.Tensor] = None


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
