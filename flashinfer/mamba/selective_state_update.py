"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..jit.mamba import gen_selective_state_update_module
from ..utils import register_custom_op, register_fake_op


@functools.cache
def get_selective_state_update_module():
    """Get cached JIT-compiled selective_state_update module.

    This uses @functools.cache for in-memory caching of loaded modules.
    The JIT system also caches compiled .so files on disk in ~/.cache/flashinfer/
    """
    return gen_selective_state_update_module().build_and_load()


@flashinfer_api
def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
) -> torch.Tensor:
    r"""Selective state update operation for Mamba layers.

    Parameters
    ----------
    state : torch.Tensor
        State tensor with shape (state_cache_size, dim, dstate) or (state_cache_size, nheads, dim, dstate)
    x : torch.Tensor
        Input tensor with shape (batch, dim) or (batch, nheads, dim)
    dt : torch.Tensor
        Delta time tensor with shape (batch, dim) or (batch, nheads, dim)
    A : torch.Tensor
        A matrix with shape (dim, dstate) or (nheads, dim, dstate)
    B : torch.Tensor
        B matrix with shape (batch, dstate) or (batch, ngroups, dstate)
    C : torch.Tensor
        C matrix with shape (batch, dstate) or (batch, ngroups, dstate)
    D : torch.Tensor
        D vector with shape (dim,) or (nheads, dim)
    z : Optional[torch.Tensor]
        Optional z tensor with shape (batch, dim) or (batch, nheads, dim)
    dt_bias : Optional[torch.Tensor]
        Optional dt bias with shape (dim,) or (nheads, dim)
    dt_softplus : bool
        Whether to apply softplus to dt
    state_batch_indices : Optional[torch.Tensor]
        Optional batch indices for cache processing
    pad_slot_id : int
        If state_batch_indices is passed, lets the kernel identify padded entries
        that will not be processed. For example: state_batch_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at indices 0 and 3

    Returns
    -------
    output : torch.Tensor
        Output tensor with shape (batch, dim) or (batch, nheads, dim)
    """
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    output = torch.empty_like(x)
    _selective_state_update(
        state,
        x,
        dt,
        output,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
    )
    return output


@register_custom_op(
    "flashinfer::selective_state_update", mutates_args=("state", "output")
)
def _selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    output: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
) -> None:
    """Internal function registered with torch.library for torch.compile() support."""
    get_selective_state_update_module().selective_state_update(
        state,
        x,
        dt,
        output,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
    )


@register_fake_op("flashinfer::selective_state_update")
def _selective_state_update_fake(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    output: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
) -> None:
    """Fake implementation for torch.compile() meta tensor propagation."""
    pass
