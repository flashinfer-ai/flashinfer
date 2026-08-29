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
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed._symmetric_memory as symm_mem

from ..api_logging import flashinfer_api
from ..jit.comm import gen_dcp_lse_reduce_module

# Keep the rendezvous handle alive and remember the group name needed by the
# C++ rendezvous lookup. The public hot-path signature remains the issue-4575
# signature: callers pass only the workspace tensor, rank, and size.
_workspace_keepalive: Dict[int, Tuple[Any, str]] = {}


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
    # 16-byte device epoch/slot metadata followed by two alternating slots.
    return 16 + 2 * cp_size * max_tokens * local_heads * (
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
    group :
        ``torch.distributed`` process group (or group name) for the CP team.

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
    symm_mem.set_backend("NCCL")
    workspace = symm_mem.empty(size_bytes, dtype=torch.uint8, device="cuda")
    workspace.zero_()
    handle = symm_mem.rendezvous(workspace, group)
    group_name = group if isinstance(group, str) else group.group_name
    _workspace_keepalive[workspace.data_ptr()] = (handle, group_name)
    return workspace


@flashinfer_api
def decode_cp_a2a_lse_reduce(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    workspace: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    is_lse_base_on_e: bool = False,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Fuse the DCP A2A exchange with the LSE-weighted reduce.

    Parameters
    ----------
    partial_o : torch.Tensor
        ``[..., cp_size, head_dim]`` CUDA tensor (fp16 or bf16).
        Typical layout: ``[batch, local_heads, cp_size, head_dim]``.
        ``partial_o[..., peer, :]`` is the slice destined for that CP rank.
    partial_lse : torch.Tensor
        ``[..., cp_size]`` CUDA float32 tensor. Leading dims must match
        ``partial_o``.
    workspace : torch.Tensor
        Rendezvoused NCCL symmetric-memory tensor from
        :func:`decode_cp_a2a_lse_reduce_create_workspace`.
    cp_rank : int
        This rank's index in the CP group.
    cp_size : int
        Context-parallel group size.
    is_lse_base_on_e : bool
        ``True`` for natural-log LSE, ``False`` for base-2 (FlashInfer MLA).
    enable_pdl : bool, optional
        Accepted for API compatibility. The ported NCCL device kernels do not
        use programmatic dependent launch.

    Returns
    -------
    torch.Tensor
        ``[..., head_dim]`` in the same dtype as ``partial_o``.
    """
    del enable_pdl
    workspace_meta = _workspace_keepalive.get(workspace.data_ptr())
    if workspace_meta is None:
        raise ValueError(
            "workspace was not created by "
            "decode_cp_a2a_lse_reduce_create_workspace"
        )
    _, group_name = workspace_meta
    get_dcp_lse_reduce_module()
    return torch.ops.flashinfer.decode_cp_a2a_lse_reduce(
        partial_o,
        partial_lse,
        workspace,
        cp_rank,
        cp_size,
        is_lse_base_on_e,
        group_name,
    )


__all__ = [
    "decode_cp_a2a_lse_reduce_workspace_size",
    "decode_cp_a2a_lse_reduce_create_workspace",
    "decode_cp_a2a_lse_reduce",
]
