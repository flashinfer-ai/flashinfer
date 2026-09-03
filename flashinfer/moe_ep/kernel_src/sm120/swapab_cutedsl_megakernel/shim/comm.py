# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared MegaMoE frontend utilities (dist bootstrap, sym heap, compile state).

Ported from ``kernel_src/cutedsl_megamoe/shim/comm.py`` (each tree carries its
own copy: the ``src.bootstrap`` / ``src.sym_buffer`` imports below must bind to
THIS tree's ``src/`` via its own ``_paths`` bootstrap).  The one SM120-specific
addition is :func:`zero_local_counter_regions` — this drop's kernel does not
tail-clean its accumulating local counters, so every re-launch must host-zero
them first (mirrors ``mega_runner._reset_local_counters``).
"""

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


def finalize_dist() -> None:
    """Tear down torch.distributed + NVSHMEM (no-op under ``MEGA_NO_DIST``)."""
    import torch.distributed as dist

    if _no_dist() or not dist.is_initialized():
        return
    from src.bootstrap import finalize_dist_and_nvshmem

    finalize_dist_and_nvshmem()


def ensure_not_capturing(what: str) -> None:
    """Fail loudly if ``what`` would run during CUDA graph stream capture.

    Guards host-side compile / symmetric-heap alloc / free paths: silently
    running them mid-capture corrupts the graph (and NVSHMEM collectives
    deadlock ranks that are not capturing). Callers place this AFTER their
    no-op early returns so a steady-state hit inside a capture stays legal.
    """
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{what} requires host-side compile/alloc/free and cannot run "
            "during CUDA graph capture. Run one eager forward on ALL EP ranks "
            "first (e.g. MoEEpMegaLayer.warmup()) so compile, workspace "
            "allocation, and autotune complete before capture."
        )


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
        # holds a stale larger peer entry.  Only the peer BASE ADDRESS is
        # needed here, and the nvshmem_ptr address mapping is deterministic,
        # so read it off the peer Buffer.
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
    # SM120 additions over the SM100 tree's _CompiledMega: the kernel's
    # combine staging is caller-allocated (no combine region inside
    # shared_workspace), the per-expert epilogue-arg tensors ride along with
    # the compile, and form A needs a compiled second-stage topk reduce.
    combine_output: Optional[torch.Tensor] = None
    combine_root: Optional[torch.Tensor] = None
    fc1_alpha: Optional[torch.Tensor] = None
    fc2_alpha: Optional[torch.Tensor] = None
    fc1_norm_const: Optional[torch.Tensor] = None
    reduce_compiled: Optional[Any] = None
    reduce_kwargs: Optional[dict] = None
    # Launch-kwargs cache: rebuilding the cute tensor views (15x from_dlpack +
    # SymBufferHost) and re-validating inputs costs real host time per launch,
    # and the launch inputs are stable session buffers in steady state.  Keyed
    # on the input data_ptrs + token count + stream; a hit skips validation
    # entirely (the same tensors were validated when the entry was built).
    # Lives here so a recompile naturally drops it.
    launch_key: Optional[tuple] = None
    launch_kwargs: Optional[dict] = None
    launch_output: Optional[torch.Tensor] = None


# The SM120 kernel does NOT reset its accumulating local counters at kernel
# tail (unlike the SM100 drop, which tail-cleans): fc1/fc2_done_counter spin
# thresholds, load_balance_counter work-stealing cursor, expert_send_count /
# l1_arrival_count / atomic_counter dispatch write-cursors, and the phase-flip
# grid_sync_counter / nvlink_barrier_counter.  A re-launch with stale values
# deadlocks or writes out of bounds, so the host zeroes exactly these regions
# before every launch (mirrors mega_runner._reset_local_counters; data buffers
# are fully overwritten each launch and deliberately NOT zeroed).  Each region
# is rank-local, so the stream-ordered zero races nothing.
_LOCAL_COUNTER_REGIONS = (
    "l1_arrival_count",
    "expert_send_count",
    "grid_sync_counter",
    "nvlink_barrier_counter",
    "fc1_done_counter",
    "fc2_done_counter",
    "atomic_counter",
    "load_balance_counter",
)


def zero_local_counter_regions(mega: _CompiledMega) -> None:
    """Zero the kernel's local flag/counter regions (pre-launch contract)."""
    kernel = mega.kernel
    for name in _LOCAL_COUNTER_REGIONS:
        if name in kernel._local_offsets:
            off = int(kernel._local_offsets[name])
            nbytes = int(kernel._local_region_by_name[name].nbytes)
            mega.local_workspace[off : off + nbytes].zero_()


def reset_compiled_mega_workspaces(mega: _CompiledMega) -> None:
    """Recovery reset after an aborted launch left the workspaces dirty.

    Multi-rank keeps the shared (peer-visible) regions untouched — zeroing
    them while a peer may still read races the symmetric heap; the local
    counter zero below plus the kernel's own overwrite-per-launch data flow
    is sufficient.  Single-rank can safely zero everything.
    """
    if getattr(mega.kernel, "world_size", 1) > 1:
        zero_local_counter_regions(mega)
    else:
        mega.shared_workspace.zero_()
        mega.local_workspace.zero_()
