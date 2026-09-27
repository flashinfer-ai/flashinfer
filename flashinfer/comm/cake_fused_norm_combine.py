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

Cake fused residual-add + RMSNorm + eight-peer BF16 combine (SM100 / SM103).

One launch per rank adds the residual to a two-track BF16 input, applies
RMSNorm per track with per-track weights, averages the two normalized tracks
into one BF16 contribution and reduces that contribution across eight peers in
ascending rank order with BF16 rounding after every addition.  Below 256
tokens the exchange is a one-shot Lamport publication (every peer polls all
eight stripes); at and above it is a fence-free two-round owner reduce (token
``t`` is owned by peer ``t % 8``).  Outputs are the normalized tracks, the
propagated residual and the replicated collective result.

Usage protocol::

    # 1. Every rank creates the peer-mapped workspace collectively
    workspace = cake_fused_norm_combine_create_workspace(
        rank=rank, world_size=8, max_tokens=2048, group=group
    )

    # 2. Launch (every rank, same token count, same launch order)
    cake_fused_norm_combine(
        x, residual, weight,
        norm_out=norm_out, residual_out=residual_out,
        collective_out=collective_out, workspace=workspace, epsilon=1e-6,
    )

    # 3. Destroy collectively after the last launch completed
    cake_fused_norm_combine_destroy_workspace(workspace)

Requirements: eight ranks on one node with CUDA IPC peer access, SM100 (B200)
or SM103 (B300) devices, hidden size 2560, two tracks, contiguous BF16 tensors
(``x``/``residual``/``norm_out``/``residual_out`` are ``[T, 2, 2560]``,
``weight`` is ``[2, 2560]``, ``collective_out`` is ``[T, 2560]``) and
``T <= max_tokens``.  A workspace carries one ordered launch sequence; every
rank must launch before any rank waits for the result.
"""

from __future__ import annotations

import ctypes
from typing import List, Literal, Optional

import torch
import torch.distributed as dist

from ..jit.cake_fused_norm_combine import (
    ARCH_BY_CAPABILITY,
    HIDDEN_DIM,
    LARGE_MIN_TOKENS,
    TRACKS,
    WIDE_MIN_TOKENS,
    WORKSPACE_TABLE_ENTRIES,
    WORLD_SIZE,
    route_applies,
    run_cake_fused_norm_combine,
    select_variant,
)
from .cuda_ipc import create_shared_buffer, cudart, free_shared_buffer
from .dlpack_utils import pack_strided_memory

# Lamport payload words are self-validating: the producer never publishes BF16
# negative zero (``0x8000``), so a cleared region is filled with that pattern.
LAMPORT_SENTINEL_U16 = 0x8000
LAMPORT_SLOTS = 3
CONTROL_WORDS = 5


def cake_fused_norm_combine_workspace_bytes(
    world_size: int, max_tokens: int, hidden: int = HIDDEN_DIM
) -> dict[str, int]:
    """Per-rank byte sizes of the three peer-mapped regions and the payload rotation."""

    if int(world_size) != WORLD_SIZE:
        raise ValueError(
            f"the Cake fused norm-combine workspace requires {WORLD_SIZE} ranks"
        )
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")
    if int(hidden) != HIDDEN_DIM:
        raise ValueError(
            f"the Cake fused norm-combine workspace requires hidden={HIDDEN_DIM}"
        )
    rotation_bytes = 2 * WORLD_SIZE * max_tokens * HIDDEN_DIM
    return {
        "data_bytes": 4 * max_tokens * HIDDEN_DIM,
        "flags_bytes": 4 * max_tokens * WORLD_SIZE,
        "rotation_bytes": rotation_bytes,
        "lamport_bytes": LAMPORT_SLOTS * rotation_bytes,
        "control_bytes": CONTROL_WORDS * 4,
        "table_entries": WORKSPACE_TABLE_ENTRIES,
    }


class CakeFusedNormCombineWorkspace:
    """Peer-mapped state of one rank: three CUDA IPC regions and the control words.

    ``workspace_ptrs`` is the ``uint64`` device table the kernels read
    (``[data_0..7, flags_0..7, lamport_0..7, control]``).  The object owns the
    control tensor and the IPC mappings; destroy it collectively with
    :func:`cake_fused_norm_combine_destroy_workspace` once every launch that
    references it has completed.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        max_tokens: int,
        hidden: int,
        group: Optional[dist.ProcessGroup],
        device: torch.device,
        data: List[int],
        flags: List[int],
        lamport: List[int],
        control: torch.Tensor,
        workspace_ptrs: torch.Tensor,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.max_tokens = max_tokens
        self.hidden = hidden
        self.group = group
        self.device = device
        self.control = control
        self.workspace_ptrs = workspace_ptrs
        self._data = data
        self._flags = flags
        self._lamport = lamport
        self._closed = False

    @property
    def rotation_bytes(self) -> int:
        return 2 * self.world_size * self.max_tokens * self.hidden

    @property
    def closed(self) -> bool:
        return self._closed

    def destroy(self) -> None:
        """Collectively release the peer mappings; every rank must call this."""

        if self._closed:
            return
        try:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            for pointers in (self._lamport, self._flags, self._data):
                free_shared_buffer(pointers, self.group)
        finally:
            self._closed = True


def cake_fused_norm_combine_create_workspace(
    *,
    rank: int,
    world_size: int,
    max_tokens: int,
    hidden: int = HIDDEN_DIM,
    group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> CakeFusedNormCombineWorkspace:
    """Collectively allocate and initialize one rank's peer-mapped workspace.

    Every rank of ``group`` must call this in the same order with the same
    ``max_tokens``.  ``device`` defaults to the current CUDA device.  The
    Lamport regions are filled with the BF16 negative-zero sentinel and the
    control words start at ``[0, 0, 0, rotation_bytes, 0]``, exactly the
    state the kernels expect before their first launch.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before creating the workspace"
        )
    if group is None:
        group = dist.group.WORLD
    if int(world_size) != dist.get_world_size(group=group) or int(
        rank
    ) != dist.get_rank(group=group):
        raise ValueError("rank/world_size must match the process group")
    sizes = cake_fused_norm_combine_workspace_bytes(world_size, max_tokens, hidden)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("device must be an explicit CUDA device")
    with torch.cuda.device(device):
        data = create_shared_buffer(sizes["data_bytes"], group)
        flags = create_shared_buffer(sizes["flags_bytes"], group)
        lamport = create_shared_buffer(sizes["lamport_bytes"], group)
        cudart.cudaMemset(ctypes.c_void_p(data[rank]), 0, sizes["data_bytes"])
        cudart.cudaMemset(ctypes.c_void_p(flags[rank]), 0, sizes["flags_bytes"])
        # FP16 negative zero shares the 0x8000 bit pattern of the BF16 sentinel.
        sentinel = pack_strided_memory(
            lamport[rank],
            sizes["lamport_bytes"],
            sizes["lamport_bytes"],
            1,
            torch.float16,
            device.index,
        )
        sentinel.fill_(-0.0)
        control = torch.tensor(
            [0, 0, 0, sizes["rotation_bytes"], 0], dtype=torch.int32, device=device
        )
        table = torch.tensor(
            [*data, *flags, *lamport, int(control.data_ptr())],
            dtype=torch.uint64,
            device=device,
        )
        torch.cuda.synchronize(device)
    dist.barrier(group=group)
    return CakeFusedNormCombineWorkspace(
        rank=int(rank),
        world_size=int(world_size),
        max_tokens=int(max_tokens),
        hidden=int(hidden),
        group=group,
        device=device,
        data=data,
        flags=flags,
        lamport=lamport,
        control=control,
        workspace_ptrs=table,
    )


def cake_fused_norm_combine_destroy_workspace(
    workspace: CakeFusedNormCombineWorkspace,
) -> None:
    """Collectively destroy a workspace created by :func:`cake_fused_norm_combine_create_workspace`."""

    if not isinstance(workspace, CakeFusedNormCombineWorkspace):
        raise TypeError("workspace must be a CakeFusedNormCombineWorkspace")
    workspace.destroy()


def _check_tensor(
    tensor: torch.Tensor, name: str, shape: tuple[int, ...], device: torch.device
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")


def cake_fused_norm_combine(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    collective_out: torch.Tensor,
    workspace: CakeFusedNormCombineWorkspace,
    epsilon: float,
    backend: Literal["cake"] = "cake",
) -> None:
    """Fused residual-add + two-track RMSNorm + eight-peer BF16 combine for this rank.

    Parameters
    ----------
    x, residual : torch.Tensor
        Contiguous BF16 ``[T, 2, 2560]`` inputs of this rank (read-only).
    weight : torch.Tensor
        Contiguous BF16 ``[2, 2560]`` per-track RMSNorm weights.
    norm_out, residual_out : torch.Tensor
        Contiguous BF16 ``[T, 2, 2560]`` caller-owned outputs.
    collective_out : torch.Tensor
        Contiguous BF16 ``[T, 2560]`` caller-owned replicated result.
    workspace : CakeFusedNormCombineWorkspace
        This rank's live workspace; ``T <= workspace.max_tokens``.
    epsilon : float
        RMSNorm epsilon.
    backend : ``"cake"``
        The only supported backend; the exported modules are the sole route.
    """

    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")
    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        raise ValueError("x must be a [tokens, tracks, hidden] tensor")
    tokens = int(x.shape[0])
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    device = x.device
    full = (tokens, TRACKS, HIDDEN_DIM)
    _check_tensor(x, "x", full, device)
    _check_tensor(residual, "residual", full, device)
    _check_tensor(weight, "weight", (TRACKS, HIDDEN_DIM), device)
    _check_tensor(norm_out, "norm_out", full, device)
    _check_tensor(residual_out, "residual_out", full, device)
    _check_tensor(collective_out, "collective_out", (tokens, HIDDEN_DIM), device)
    if not isinstance(workspace, CakeFusedNormCombineWorkspace):
        raise TypeError("workspace must be a CakeFusedNormCombineWorkspace")
    if workspace.closed:
        raise RuntimeError("the Cake fused norm-combine workspace was destroyed")
    if workspace.device != device:
        raise ValueError("tensors must live on the workspace device")
    if tokens > workspace.max_tokens:
        raise ValueError(
            f"tokens ({tokens}) exceed workspace.max_tokens ({workspace.max_tokens})"
        )
    if not route_applies(
        world_size=workspace.world_size,
        device_capability=tuple(torch.cuda.get_device_capability(device)),
        hidden_dim=workspace.hidden,
    ):
        raise ValueError(
            "the Cake fused norm-combine export covers eight SM100 or SM103 peers with hidden size 2560"
        )
    run_cake_fused_norm_combine(
        backend="cake",
        x=x,
        residual=residual,
        weight=weight,
        norm_out=norm_out,
        residual_out=residual_out,
        collective_out=collective_out,
        workspace_ptrs=workspace.workspace_ptrs,
        rank=workspace.rank,
        tokens=tokens,
        epsilon=float(epsilon),
    )


__all__ = [
    "ARCH_BY_CAPABILITY",
    "CONTROL_WORDS",
    "HIDDEN_DIM",
    "LAMPORT_SENTINEL_U16",
    "LAMPORT_SLOTS",
    "LARGE_MIN_TOKENS",
    "TRACKS",
    "WIDE_MIN_TOKENS",
    "WORLD_SIZE",
    "CakeFusedNormCombineWorkspace",
    "cake_fused_norm_combine",
    "cake_fused_norm_combine_create_workspace",
    "cake_fused_norm_combine_destroy_workspace",
    "cake_fused_norm_combine_workspace_bytes",
    "select_variant",
]
