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
)
from ..utils import get_compute_capability, register_custom_op, register_fake_op


@functools.cache
def _get_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    sm_major: int,
    state_scale_dtype: Optional[torch.dtype] = None,
    philox_rounds: int = 0,
):
    args = (
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
        state_scale_dtype,
        philox_rounds,
    )
    if sm_major >= 9:
        return gen_selective_state_update_sm90_module(*args).build_and_load()
    else:
        return gen_selective_state_update_module(*args).build_and_load()


def get_selective_state_update_module(
    device: torch.device,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    state_scale_dtype: Optional[torch.dtype] = None,
    philox_rounds: int = 0,
):
    major, _ = get_compute_capability(device)
    return _get_module(
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
        major,
        state_scale_dtype,
        philox_rounds,
    )


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
    state_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    intermediate_state_scales: Optional[torch.Tensor] = None,
    rand_seed: Optional[int] = None,
    philox_rounds: int = 10,
    cache_steps: int = 0,
    algorithm: str = "auto",
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
    state_scale : Optional[torch.Tensor]
        Optional float32 scale tensor with shape (state_cache_size, nheads, dim)
        for int16 state quantization with block scaling
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
    rand_seed : Optional[int]
        Optional integer seed for stochastic rounding (Philox-4x32 PRNG).
        When provided, state values are stochastically rounded before storing to fp16.
        When None, no stochastic rounding is applied (regardless of philox_rounds).
        Cannot be used together with state_scale.
    philox_rounds : int
        Number of Philox-4x32 PRNG rounds for stochastic rounding (default 10,
        matching Triton's tl.randint). Only effective when rand_seed is not None;
        ignored otherwise. Must be non-negative.
    cache_steps : int
        Number of steps/tokens to cache for speculative decoding
    algorithm : str
        Algorithm to use: "auto" (default, picks the best kernel based on GPU arch,
        data types, and problem size), "simple" (all GPUs), "vertical" and "horizontal"
        (SM90+ only). MTP mode only supports "auto" or "simple".

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
    # Normalize state_scale to 3D: (state_cache_size, nheads, dim)
    if state_scale is not None and state_scale.dim() == 4 and state_scale.size(-1) == 1:
        state_scale = state_scale.squeeze(-1)

    # Validate rand_seed and philox_rounds
    if rand_seed is not None:
        assert isinstance(rand_seed, int), "rand_seed must be an integer"
        assert state_scale is None, "rand_seed and state_scale cannot both be provided"
        assert philox_rounds > 0, "philox_rounds must be > 0 when rand_seed is provided"
    else:
        # No stochastic rounding when rand_seed is None
        philox_rounds = 0
        rand_seed = 0

    if out is None:
        output = torch.empty_like(x)
    else:
        output = out

    # Determine stateIndex dtype from index tensors, default to int32
    stateIndex_dtype = torch.int32
    if state_batch_indices is not None:
        stateIndex_dtype = state_batch_indices.dtype
    elif intermediate_state_indices is not None:
        stateIndex_dtype = intermediate_state_indices.dtype

    # Extract dim/dstate/ntokens for JIT specialization
    dim = state.size(2)
    dstate = state.size(3)
    ntokens_mtp = x.size(1) if x.dim() == 4 else 1

    if algorithm == "auto":
        algorithm_int = 0
    elif algorithm == "simple":
        algorithm_int = 1
    elif algorithm == "vertical":
        algorithm_int = 2
    elif algorithm == "horizontal":
        algorithm_int = 3
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

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
        state_scale,
        output,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        intermediate_state_scales,
        rand_seed,
        cache_steps,
        algorithm_int,
        philox_rounds,
        state.dtype,
        x.dtype,
        dt.dtype,
        A.dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
    )
    return output


@register_custom_op(
    "flashinfer::selective_state_update",
    mutates_args=(
        "state",
        "output",
        "intermediate_states_buffer",
        "state_scale",
        "intermediate_state_scales",
    ),
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
    state_scale: Optional[torch.Tensor],
    output: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: int,
    cache_steps: int,
    algorithm: int,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
) -> None:
    """Internal function registered with torch.library for torch.compile() support."""
    get_selective_state_update_module(
        state.device,
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
        state_scale_dtype=state_scale.dtype if state_scale is not None else None,
        philox_rounds=philox_rounds,
    ).selective_state_update(
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
        state_scale,
        output,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        intermediate_state_scales,
        rand_seed,
        cache_steps,
        algorithm,
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
    state_scale: Optional[torch.Tensor],
    output: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: int,
    cache_steps: int,
    algorithm: int,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
) -> None:
    """Fake implementation for torch.compile() meta tensor propagation."""
    pass
