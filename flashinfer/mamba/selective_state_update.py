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
from ..jit.mamba import (
    gen_selective_state_update_module,
    gen_selective_state_update_sm90_module,
    gen_selective_state_update_sm100_module,
)
from ..utils import get_compute_capability, register_custom_op, register_fake_op


@functools.cache
def get_selective_state_update_module_base():
    """Get cached JIT-compiled selective_state_update module (base version)."""
    return gen_selective_state_update_module().build_and_load()


@functools.cache
def get_selective_state_update_module_sm90():
    """Get cached JIT-compiled selective_state_update module (SM90/Hopper version)."""
    return gen_selective_state_update_sm90_module().build_and_load()


@functools.cache
def get_selective_state_update_module_sm100():
    """Get cached JIT-compiled selective_state_update module (SM100+/Blackwell version)."""
    return gen_selective_state_update_sm100_module().build_and_load()


def get_selective_state_update_module(device: torch.device):
    major, _ = get_compute_capability(device)
    if major >= 10:
        # SM100+ (Blackwell and newer) uses horizontal producer-consumer kernel
        return get_selective_state_update_module_sm100()
    elif major == 9:
        # SM90 (Hopper) uses vertical producer-consumer kernel
        return get_selective_state_update_module_sm90()
    else:
        # Pre-Hopper uses simple kernel
        return get_selective_state_update_module_base()


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
    out: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: int = 0,
) -> torch.Tensor:
    r"""Selective state update operation for Mamba layers (the generation phase).

    Parameters
    ----------
    state : torch.Tensor
        State tensor with shape (state_cache_size, dim, dstate) or (state_cache_size, nheads, dim, dstate)
    x : torch.Tensor
        Input tensor with shape (batch, dim) or (batch, nheads, dim) for single-token
        or (batch, T, nheads, dim) for multi-token
    dt : torch.Tensor
        Delta time tensor with shape (batch, dim) or (batch, nheads, dim) for single-token
        or (batch, T, nheads, dim) for multi-token
    A : torch.Tensor
        A matrix with shape (dim, dstate) or (nheads, dim, dstate)
    B : torch.Tensor
        B matrix with shape (batch, dstate) or (batch, ngroups, dstate) for single-token
        or (batch, T, ngroups, dstate) for multi-token
    C : torch.Tensor
        C matrix with shape (batch, dstate) or (batch, ngroups, dstate) for single-token
        or (batch, T, ngroups, dstate) for multi-token
    D : torch.Tensor
        D vector with shape (dim,) or (nheads, dim)
    z : Optional[torch.Tensor]
        Optional z tensor with shape (batch, dim) or (batch, nheads, dim) for single-token
        or (batch, T, nheads, dim) for multi-token
    dt_bias : Optional[torch.Tensor]
        Optional dt bias with shape (dim,) or (nheads, dim)
    dt_softplus : bool
        Whether to apply softplus to dt
    state_batch_indices : Optional[torch.Tensor]
        Optional batch indices for cache processing with shape (batch,)
    pad_slot_id : int
        If state_batch_indices is passed, lets the kernel identify padded entries
        that will not be processed. For example: state_batch_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at indices 0 and 3
    out : Optional[torch.Tensor]
        Optional output tensor (same shape as x)
    disable_state_update : bool
        If True, skip updating the state tensor (useful for speculative decoding verification)
    intermediate_states_buffer : Optional[torch.Tensor]
        Optional buffer for caching intermediate states during speculative decoding
        with shape (batch, cache_steps, nheads, dim, dstate)
    intermediate_state_indices : Optional[torch.Tensor]
        Optional indices mapping batch elements to intermediate state buffer positions
        with shape (batch,)
    cache_steps : int
        Number of steps/tokens to cache for speculative decoding

    Returns
    -------
    output : torch.Tensor
        Output tensor with shape (batch, dim) or (batch, nheads, dim) for single-token
        or (batch, T, nheads, dim) for multi-token
    """
    # Determine if we're in multi-token mode (more than 1 token)
    is_mtp = cache_steps >= 1

    if state.dim() == 3:
        state = state.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if D.dim() == 1:
        D = D.unsqueeze(0)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    # Handle x, dt, B, C, z dimensions based on mode
    # For single-token: 2D -> 3D (batch, nheads, dim)
    # For multi-token: 3D -> 4D (batch, T, nheads, dim)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if is_mtp and x.dim() == 3:
        # Add T dimension for MTP mode: (batch, nheads, dim) -> (batch, T, nheads, dim)
        x = x.unsqueeze(1)

    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if is_mtp and dt.dim() == 3:
        dt = dt.unsqueeze(1)

    if B.dim() == 2:
        B = B.unsqueeze(1)
    if is_mtp and B.dim() == 3:
        B = B.unsqueeze(1)

    if C.dim() == 2:
        C = C.unsqueeze(1)
    if is_mtp and C.dim() == 3:
        C = C.unsqueeze(1)

    if z is not None:
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if is_mtp and z.dim() == 3:
            z = z.unsqueeze(1)
    if out is None:
        output = torch.empty_like(x)
    else:
        output = out
    _selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        output,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
    )
    return output


@register_custom_op(
    "flashinfer::selective_state_update",
    mutates_args=("state", "output", "intermediate_states_buffer"),
)
def _selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    output: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    cache_steps: int,
) -> None:
    """Internal function registered with torch.library for torch.compile() support."""
    get_selective_state_update_module(state.device).selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        output,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
    )


@register_fake_op("flashinfer::selective_state_update")
def _selective_state_update_fake(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    output: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    cache_steps: int,
) -> None:
    """Fake implementation for torch.compile() meta tensor propagation."""
    pass
