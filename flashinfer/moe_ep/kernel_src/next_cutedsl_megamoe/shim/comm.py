# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Symmetric-heap + graph-capture utilities for the SM107 GLU mega frontend.

Mirrors ``kernel_src/cutedsl_megamoe/shim/comm.py`` (the SM100 tree) with the
drop-specific ``src.bootstrap`` pieces removed: torch.distributed / NVSHMEM
runtime bring-up for this tree is owned by ``flashinfer.moe_ep.core.runtime``
(the backend declares the requirement); the shim only allocates on whatever
heap is already up.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch


def _no_dist() -> bool:
    # Read at call time, not import time: callers (e.g. single-rank pytest
    # tests) set MEGA_NO_DIST=1 after this module is already imported.
    return bool(int(os.environ.get("MEGA_NO_DIST", "0")))


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
            "first so compile and workspace allocation complete before capture."
        )


def sym_zeros(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Zero-initialised symmetric-heap tensor (plain CUDA when ``MEGA_NO_DIST=1``).

    Always allocated as a flat uint8 root and viewed to ``dtype``: nvshmem4py's
    dtype table has no fp8 entries, and torch.zeros's fill_cuda is
    unimplemented for some fp8 dtypes (matches the SM100 tree's
    ``_sym_zeros_byte_view_1b`` convention). The root rides on the view so
    :func:`free_sym_tensor` can release the actual allocation.
    """
    nbytes = torch.empty(0, dtype=dtype).element_size()
    for dim in shape:
        nbytes *= dim
    if _no_dist():
        root = torch.zeros((max(nbytes, 1),), dtype=torch.uint8, device="cuda")
        tensor = root[:nbytes].view(dtype).reshape(shape)
        # Tag so free_sym_tensor frees by allocation kind, not by whatever
        # MEGA_NO_DIST happens to be at free time (the env can be flipped
        # back between alloc and free, e.g. by pytest monkeypatch teardown).
        tensor._mega_plain_alloc = True
        return tensor
    import nvshmem.core

    root = nvshmem.core.tensor((max(nbytes, 1),), dtype=torch.uint8)
    root.zero_()
    tensor = root[:nbytes].view(dtype).reshape(shape)
    tensor._mega_sym_root = root
    return tensor


def free_sym_tensor(tensor: Optional[torch.Tensor]) -> None:
    """Release an NVSHMEM symmetric tensor; no-op for plain-CUDA allocations."""
    if tensor is None or getattr(tensor, "_mega_plain_alloc", False) or _no_dist():
        return
    import nvshmem.core

    try:
        nvshmem.core.free_tensor(getattr(tensor, "_mega_sym_root", tensor))
    except (RuntimeError, ValueError, TypeError) as exc:
        msg = str(exc).lower()
        if any(token in msg for token in ("already", "freed", "invalid")):
            return
        raise


def compute_peer_offsets(
    sym_tensor: torch.Tensor,
    world_size: int,
) -> Tuple[int, Tuple[int, ...]]:
    """(local base address, per-peer base offsets) for a symmetric tensor.

    NVSHMEM lays the heap out identically on every rank, so the
    ``peer_base - local_base`` delta computed from ONE symmetric allocation is
    valid for EVERY allocation on the heap — the device side resolves a peer
    copy of any pointer as ``ptr + offsets[peer]``.
    """
    local_base = int(sym_tensor.data_ptr())
    if _no_dist():
        return local_base, tuple(0 for _ in range(world_size))
    import nvshmem.core
    from nvshmem.core.interop.torch import tensor_get_buffer

    my_pe = int(nvshmem.core.my_pe())
    # Unwrap byte-view allocations to their root: the nvshmem4py tracker is
    # keyed on the actual allocation, not torch views of it.
    buf, _size, _dtype = tensor_get_buffer(
        getattr(sym_tensor, "_mega_sym_root", sym_tensor)
    )

    def _peer_base(peer: int) -> int:
        # Own rank maps to the local base (nvshmem_ptr identity). Skipping the
        # get_peer_buffer call for it avoids nvshmem4py's self-peer tracker
        # ref-count bump, which defers the real free to GC (see the SM100
        # shim's comm.py for the full history).
        if peer == my_pe:
            return local_base
        # get_peer_buffer (not get_peer_tensor): only the peer BASE ADDRESS is
        # needed, and get_peer_tensor's .view(shape) breaks on stale tracker
        # entries when the heap reuses an address for a smaller allocation.
        peer_buf = nvshmem.core.get_peer_buffer(buf, peer)
        return int(torch.utils.dlpack.from_dlpack(peer_buf).data_ptr())

    return local_base, tuple(
        _peer_base(peer) - local_base for peer in range(world_size)
    )
