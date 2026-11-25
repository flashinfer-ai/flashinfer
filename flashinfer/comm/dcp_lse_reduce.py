"""Fused decode-CP all-to-all + LSE-weighted reduce.

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

Public names follow https://github.com/flashinfer-ai/flashinfer/issues/4575.

The implementation is self-contained in FlashInfer and uses NCCL's device-side
LSA APIs over a tensor allocated by torch's NCCL symmetric-memory backend. It
does not call ``libnccl_longseq`` and does not use the Helix/MNNVL A2A kernel.
"""

import functools
import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
import torch.distributed._symmetric_memory as symm_mem

from ..api_logging import flashinfer_api
from ..jit.comm import gen_dcp_lse_reduce_module
from ..trace.templates.comm import decode_cp_a2a_lse_reduce_trace_dispatch


@dataclass
class _WorkspaceState:
    """Python-owned state associated with one symmetric-memory workspace."""

    handle: Any
    group_name: str
    device: torch.device
    stream_id: Optional[int] = None


# Keep the rendezvous handle alive and remember the group name needed by the
# C++ rendezvous lookup. The public hot-path signature remains the issue-4575
# signature: callers pass only the workspace tensor, rank, and size.
_workspace_keepalive: Dict[int, _WorkspaceState] = {}


def _release_workspace(data_ptr: int, state: _WorkspaceState) -> None:
    """Release workspace state after its queued CUDA work has completed."""
    if _workspace_keepalive.get(data_ptr) is not state:
        return
    try:
        torch.cuda.synchronize(state.device)
    except (AttributeError, RuntimeError):
        # Finalizers may run while CUDA is shutting down. Removing the matching
        # entry still avoids retaining a stale rendezvous handle indefinitely.
        pass
    finally:
        if _workspace_keepalive.get(data_ptr) is state:
            del _workspace_keepalive[data_ptr]


def _get_workspace_state(workspace: torch.Tensor) -> _WorkspaceState:
    state = _workspace_keepalive.get(workspace.data_ptr())
    if state is None:
        raise ValueError(
            "workspace was not created by decode_cp_a2a_lse_reduce_create_workspace"
        )

    # Raw-stream lookup avoids constructing a torch.cuda.Stream on every call.
    stream_id = torch._C._cuda_getCurrentRawStream(workspace.device.index)
    if state.stream_id is None:
        state.stream_id = stream_id
    elif state.stream_id != stream_id:
        raise RuntimeError(
            "a DCP LSE reduce workspace may only be used from one ordered CUDA "
            "stream; allocate a separate workspace for each concurrent stream"
        )
    return state


def _dcp_lse_reduce_payload_offset(cp_size: int) -> int:
    # 16-byte epoch metadata plus two uint32 readiness arrays, rounded so the
    # vectorized payload remains 16-byte aligned.
    return (16 + 2 * cp_size * 4 + 15) // 16 * 16


@functools.cache
def get_dcp_lse_reduce_module():
    """Build and load the torch custom op once."""
    return gen_dcp_lse_reduce_module().build_and_load()


def decode_cp_a2a_lse_reduce_workspace_size(
    max_tokens: int,
    local_heads: int,
    cp_size: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Return the required NCCL symmetric workspace size in bytes."""
    if min(max_tokens, local_heads, cp_size, head_dim) <= 0:
        raise ValueError("workspace geometry must be positive")
    if cp_size > 64:
        raise ValueError("cp_size must not exceed 64")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("dtype must be torch.float16 or torch.bfloat16")
    itemsize = torch.empty((), dtype=dtype).element_size()
    if head_dim * itemsize % 16 != 0:
        raise ValueError("head_dim rows must be 16-byte aligned")
    payload_offset = _dcp_lse_reduce_payload_offset(cp_size)
    return payload_offset + 2 * cp_size * max_tokens * local_heads * (
        head_dim * itemsize + 4
    )


@flashinfer_api
def decode_cp_a2a_lse_reduce_create_workspace(
    max_tokens: int,
    local_heads: int,
    cp_size: int,
    head_dim: int,
    dtype: torch.dtype,
    group: Any,
) -> torch.Tensor:
    """Create and rendezvous the fused op's NCCL symmetric workspace.

    All ranks in ``group`` must call this function collectively. The group must
    fit in one NCCL load/store-accessible (LSA) NVLink domain; multi-node groups
    spanning LSA domains are not supported. Allocate one workspace per group
    and reuse it for every invocation and CUDA graph replay. A workspace may
    only be used from one ordered CUDA stream; allocate a workspace per
    concurrent stream.

    Parameters
    ----------
    max_tokens : int
        Upper bound on the token/batch dimension of later calls.
    local_heads : int
        Number of heads this rank keeps after the reduce-scatter.
    cp_size : int
        Context-parallel group size.
    head_dim : int
        Elements per head.
    dtype : torch.dtype
        Storage type of ``partial_o`` / ``output`` (fp16 or bf16).
    group : torch.distributed.ProcessGroup or str
        Process group (or group name) for the CP team.

    Returns
    -------
    torch.Tensor
        A rendezvoused NCCL symmetric-memory ``uint8`` tensor. Allocate it once
        before CUDA graph capture and reuse it.
    """
    # Build/load on every rank before entering either NCCL collective below.
    # The JIT cache lock may otherwise leave one rank creating the devcomm
    # while another rank is still compiling.
    get_dcp_lse_reduce_module()
    size_bytes = decode_cp_a2a_lse_reduce_workspace_size(
        max_tokens, local_heads, cp_size, head_dim, dtype
    )
    group_name = group if isinstance(group, str) else group.group_name
    symm_mem.set_backend("NCCL")
    # PyTorch's NCCL symmetric-memory communicator registry is keyed by the
    # concrete CUDA device. Do not use the unindexed ``"cuda"`` device here:
    # it would rendezvous through a separate registry entry from the process
    # group's ``cuda:<local_rank>`` communicator.
    device = torch.device("cuda", torch.cuda.current_device())
    workspace = symm_mem.empty(size_bytes, dtype=torch.uint8, device=device)
    # Initialize the local epoch and readiness words. The payload is fully
    # overwritten before every read; clearing it could race a peer's first put.
    workspace[: _dcp_lse_reduce_payload_offset(cp_size)].zero_()
    # Initialization must complete before rendezvous; afterwards any peer may
    # enter the first fused kernel and publish a remote readiness value.
    torch.cuda.current_stream().synchronize()
    handle = symm_mem.rendezvous(workspace, group)
    state = _WorkspaceState(handle=handle, group_name=group_name, device=device)
    data_ptr = workspace.data_ptr()
    _workspace_keepalive[data_ptr] = state
    weakref.finalize(workspace, _release_workspace, data_ptr, state)
    return workspace


@flashinfer_api(trace=decode_cp_a2a_lse_reduce_trace_dispatch)
def decode_cp_a2a_lse_reduce(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    workspace: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    lse_mode: Literal["base2", "basee"] = "base2",
    *,
    mask: Optional[torch.Tensor] = None,
    layout: Optional[Literal["dst_major", "head_major"]] = None,
) -> torch.Tensor:
    """Fuse an NCCL LSA DCP A2A exchange with the LSE-weighted reduce.

    Send, receive synchronization, and reduction execute in one cooperative
    CUDA kernel.

    This is a collective operation. Every rank must invoke it in the same order
    and the same number of times, including CUDA graph replays. All ranks must
    belong to one NCCL LSA/NVLink domain.

    Parameters
    ----------
    partial_o : torch.Tensor
        Contiguous ``[batch, cp_size, heads, head_dim]`` CUDA tensor (fp16 or
        bf16), or more generally ``[..., cp_size, heads, head_dim]``.
        ``partial_o[..., peer, :, :]`` is the slice destined for that CP rank.
        The data pointer and each head row must be 16-byte aligned.
        Destination-major ``[batch, cp_size * heads, head_dim]`` attention
        output can be viewed with ``unflatten(1, (cp_size, heads))``.
        The head-major ``[..., heads, cp_size, head_dim]`` order accepted
        through v0.7.5 is deprecated; see ``layout``.
    partial_lse : torch.Tensor
        Contiguous ``[batch, cp_size, heads]`` CUDA float32 tensor, or more
        generally ``[..., cp_size, heads]``. Its shape must match
        ``partial_o.shape[:-1]``. Each destination receives ``heads`` outputs;
        this operation does not split the head axis further.
    workspace : torch.Tensor
        Rendezvoused NCCL symmetric-memory tensor from
        :func:`decode_cp_a2a_lse_reduce_create_workspace`. Reuse it only from
        one ordered CUDA stream.
    cp_rank : int
        This rank's index in the CP group.
    cp_size : int
        Context-parallel group size.
    lse_mode : Literal["base2", "basee"]
        Logarithm base used by ``partial_lse``. ``"base2"`` is the default
        produced by FlashInfer MLA; ``"basee"`` selects natural-log LSE.

    mask : Optional[torch.Tensor]
        Optional contiguous CUDA bool ``[batch]`` on the input device, supported
        for 4-D ``partial_o`` and 3-D ``partial_lse``. Each entry marks a token's
        validity on this source CP rank, shared across heads and destinations.
        False skips reading its O/LSE and sends O=0 and LSE=-inf while still
        participating in collective synchronization. ``None`` disables token
        masking. Nonfinite LSE also excludes its O without reading it; an
        all-excluded row returns zero. Finite LSE with NaN O retains NaN
        propagation. For CUDA graphs, keep mask storage alive and update its
        contents on the ordered stream.
    layout : Optional[Literal["dst_major", "head_major"]]
        Order of the ``cp_size`` and ``heads`` axes in ``partial_o`` and
        ``partial_lse``. ``"dst_major"`` is ``[..., cp_size, heads, head_dim]``.
        ``"head_major"`` is the deprecated ``[..., heads, cp_size, head_dim]``
        order (``[..., heads, cp_size]`` for ``partial_lse``, and also
        ``[cp_size, head_dim]`` with no head axis); such inputs are copied into
        destination-major order and raise a ``DeprecationWarning``. Head-major
        support is removed after two minor releases (v0.7.8).
        ``None`` infers the order from which axis has length ``cp_size``. When
        both do (``heads == cp_size > 1``) the order cannot be inferred and
        ``layout`` must be given.

    Returns
    -------
    torch.Tensor
        ``[batch, heads, head_dim]``, or more generally ``[..., heads, head_dim]``,
        in the same dtype as ``partial_o``.
    """
    if lse_mode not in ("base2", "basee"):
        raise ValueError(f"lse_mode must be 'base2' or 'basee', got {lse_mode!r}")
    squeeze = False
    if _resolve_layout(partial_o, cp_size, layout) == "head_major":
        warnings.warn(
            "decode_cp_a2a_lse_reduce: head-major [..., heads, cp_size, head_dim] "
            "inputs are deprecated and copied into destination-major order; pass "
            "[..., cp_size, heads, head_dim] (layout='dst_major'). Head-major "
            "support is removed after two minor releases (v0.7.8).",
            DeprecationWarning,
            stacklevel=3,
        )
        if partial_lse.shape != partial_o.shape[:-1]:
            raise ValueError(
                f"partial_lse shape {tuple(partial_lse.shape)} must equal "
                f"partial_o.shape[:-1] = {tuple(partial_o.shape[:-1])}"
            )
        # A 2-D [cp_size, head_dim] input has no head axis; give it one.
        squeeze = partial_o.dim() == 2
        if squeeze:
            partial_o = partial_o.unsqueeze(0)
            partial_lse = partial_lse.unsqueeze(0)
        partial_o = partial_o.transpose(-3, -2).contiguous()
        partial_lse = partial_lse.transpose(-2, -1).contiguous()
    group_name = _get_workspace_state(workspace).group_name
    get_dcp_lse_reduce_module()
    out = torch.ops.flashinfer.decode_cp_a2a_lse_reduce(
        partial_o,
        partial_lse,
        workspace,
        cp_rank,
        cp_size,
        lse_mode == "basee",
        group_name,
        mask,
    )
    return out.squeeze(0) if squeeze else out


def _resolve_layout(
    partial_o: torch.Tensor, cp_size: int, layout: Optional[str]
) -> str:
    """Return ``"dst_major"`` or ``"head_major"`` for this call's inputs."""
    if layout is not None:
        if layout not in ("dst_major", "head_major"):
            raise ValueError(
                f"layout must be 'dst_major', 'head_major' or None, got {layout!r}"
            )
        return layout
    # One .shape read: each Tensor.dim()/size() call is a C++ round trip, and
    # this runs on every eager decode call.
    shape = partial_o.shape
    ndim = len(shape)
    if ndim == 2:
        return "head_major"  # only the head-major contract had 2-D inputs
    if ndim < 2:
        return "dst_major"  # let the kernel's shape checks report it
    dst = shape[-3] == cp_size
    head = shape[-2] == cp_size
    if dst and head and cp_size > 1:
        raise ValueError(
            "decode_cp_a2a_lse_reduce cannot infer the input layout because "
            f"heads == cp_size == {cp_size}; pass layout='dst_major' for "
            "[..., cp_size, heads, head_dim] or layout='head_major' for the "
            "deprecated [..., heads, cp_size, head_dim]."
        )
    return "head_major" if head and not dst else "dst_major"


__all__ = [
    "decode_cp_a2a_lse_reduce_workspace_size",
    "decode_cp_a2a_lse_reduce_create_workspace",
    "decode_cp_a2a_lse_reduce",
]
