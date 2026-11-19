# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Symmetric allocation and CUDA-graph helpers.

Peer-written storage comes from PyTorch symmetric memory: each allocation is
rendezvoused over the EP process group, and the kernel reaches a peer's copy of
any byte of an allocation as ``local address + offsets[peer]``.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch


def _no_dist() -> bool:
    # Read at call time, not import time: single-rank tests set MEGA_NO_DIST=1
    # after this module is imported.
    return bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def ensure_not_capturing(what: str) -> None:
    """Fail loudly if ``what`` would run during CUDA graph stream capture.

    Guards host-side compile and symmetric alloc/free paths: running them
    mid-capture corrupts the graph, and the collective rendezvous deadlocks ranks
    that are not capturing. Callers place this after their no-op early returns so
    a steady-state hit inside a capture stays legal.
    """
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{what} requires host-side compile/alloc/free and cannot run "
            "during CUDA graph capture. Run one eager forward on ALL EP ranks "
            "first so compile and workspace allocation complete before capture."
        )


def sym_zeros(shape: Tuple[int, ...], dtype: torch.dtype, group=None) -> torch.Tensor:
    """Zero-initialized symmetric tensor (plain CUDA memory when ``MEGA_NO_DIST=1``).

    Collective over ``group`` (default: the world group). Always allocated as a
    flat uint8 root and viewed to ``dtype``, so FP8 dtypes need no fill support.
    Returns only after every rank has zeroed its copy, so no peer can write into
    memory that is still being initialized.
    """
    nbytes = torch.empty(0, dtype=dtype).element_size()
    for dim in shape:
        nbytes *= dim
    if _no_dist():
        root = torch.zeros((max(nbytes, 1),), dtype=torch.uint8, device="cuda")
        tensor = root[:nbytes].view(dtype).reshape(shape)
        tensor._mega_sym_root = root
        return tensor
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    group = group or dist.group.WORLD
    root = symm_mem.empty(max(nbytes, 1), dtype=torch.uint8, device="cuda")
    handle = symm_mem.rendezvous(root, group)
    root.zero_()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    tensor = root[:nbytes].view(dtype).reshape(shape)
    tensor._mega_sym_root = root
    tensor._mega_symm_handle = handle
    return tensor


def sym_zeros_views(
    specs: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    group=None,
    byte_alignment: int = 128,
) -> List[torch.Tensor]:
    """Zero-initialized tensors carved out of one symmetric allocation.

    Each ``(shape, dtype)`` gets a ``byte_alignment``-aligned span of the same
    root, so one peer offset table (``compute_peer_offsets`` of any of them)
    reaches every one of them. Collective over ``group`` like ``sym_zeros``.
    """
    spans = []
    total_bytes = 0
    for shape, dtype in specs:
        nbytes = torch.empty(0, dtype=dtype).element_size()
        for dim in shape:
            nbytes *= dim
        total_bytes = -(-total_bytes // byte_alignment) * byte_alignment
        spans.append((total_bytes, nbytes))
        total_bytes += nbytes
    flat = sym_zeros((max(total_bytes, 1),), torch.uint8, group)
    views = []
    for (shape, dtype), (start, nbytes) in zip(specs, spans, strict=True):
        view = flat[start : start + nbytes].view(dtype).reshape(shape)
        view._mega_sym_root = flat._mega_sym_root
        if hasattr(flat, "_mega_symm_handle"):
            view._mega_symm_handle = flat._mega_symm_handle
        views.append(view)
    return views


def free_sym_tensor(tensor: Optional[torch.Tensor]) -> None:
    """Drop the allocation's handle; the memory is released with its last reference."""
    if tensor is not None and hasattr(tensor, "_mega_symm_handle"):
        tensor._mega_symm_handle = None


def compute_peer_offsets(
    sym_tensor: torch.Tensor,
    world_size: int,
) -> Tuple[int, Tuple[int, ...]]:
    """(local base address, per-peer byte offsets) of a symmetric tensor.

    The offsets are only valid for addresses inside ``sym_tensor``'s allocation.
    The own rank's offset is zero: a rank touches its own copy through one
    virtual address only.
    """
    local_base = int(sym_tensor.data_ptr())
    if _no_dist():
        return local_base, tuple(0 for _ in range(world_size))
    handle = sym_tensor._mega_symm_handle
    pointers = [int(pointer) for pointer in handle.buffer_ptrs]
    if pointers[handle.rank] != int(sym_tensor._mega_sym_root.data_ptr()):
        raise RuntimeError(
            "symmetric buffer self entry does not match the local allocation"
        )
    return local_base, tuple(
        pointers[peer] - pointers[handle.rank] for peer in range(world_size)
    )
