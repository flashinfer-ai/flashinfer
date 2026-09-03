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
"""

import functools
from typing import Optional
import torch
from ..api_logging import flashinfer_api
from ..jit.mamba.replayssm_materialize import gen_replayssm_materialize_module


@functools.cache
def _module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    dim: int,
    dstate: int,
    heads_per_group: int,
    max_window: int,
    philox_rounds: int,
):
    return gen_replayssm_materialize_module(
        state_dtype,
        input_dtype,
        matrixA_dtype,
        dim,
        dstate,
        heads_per_group,
        max_window,
        philox_rounds,
    ).build_and_load()


@flashinfer_api
def replayssm_materialize(
    state_ptrs: torch.Tensor,
    state_slot_strides: torch.Tensor,
    x_cache_ptrs: torch.Tensor,
    x_cache_slot_strides: torch.Tensor,
    B_cache_ptrs: torch.Tensor,
    B_cache_slot_strides: torch.Tensor,
    dt_cache_ptrs: torch.Tensor,
    dt_cache_slot_strides: torch.Tensor,
    A_ptrs: torch.Tensor,
    state_scale_ptrs: torch.Tensor,
    state_scale_slot_strides: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    ring_start: torch.Tensor,
    replay_prefix_len: torch.Tensor,
    active_request_indices: torch.Tensor,
    *,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    dim: int,
    dstate: int,
    num_heads: int,
    heads_per_group: int,
    max_window: int,
    ring_buffer_len: int,
    pad_slot_id: int = -1,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 0,
) -> None:
    """Materialize an SSM state at a selected token, from an older state and a
    ReplaySSM ring buffer.

    A source state plus its replay ring represents every state from the source
    state through the state advanced by the accepted-token prefix, see
    :func:`flashinfer.mamba.checkpointing_ssu`. This operation starts at
    ``ring_start``, applies a caller-selected prefix of that ring to the source
    state, and writes the resulting state to a separate destination slot. It
    does not consume or otherwise modify replay-ring data.

    Parameters
    ----------
    state_ptrs : torch.Tensor
        One-dimensional CUDA int64 pointer table of shape ``(L,)``. Each
        entry is the base address of that layer's state storage.
    state_slot_strides : torch.Tensor
        One-dimensional CUDA int64 table of shape ``(L,)`` giving the state
        storage outer-slot stride, in elements, for each layer.
    x_cache_ptrs : torch.Tensor
        One-dimensional CUDA int64 pointer table of shape ``(L,)`` for the
        per-head replay ``x`` cache of each layer.
    x_cache_slot_strides : torch.Tensor
        One-dimensional CUDA int64 table of shape ``(L,)`` giving the ``x``
        cache outer-slot stride, in elements, for each layer.
    B_cache_ptrs : torch.Tensor
        One-dimensional CUDA int64 pointer table of shape ``(L,)`` for the
        per-group replay ``B`` cache of each layer.
    B_cache_slot_strides : torch.Tensor
        One-dimensional CUDA int64 table of shape ``(L,)`` giving the ``B``
        cache outer-slot stride, in elements, for each layer.
    dt_cache_ptrs : torch.Tensor
        One-dimensional CUDA int64 pointer table of shape ``(L,)`` for the
        replay ``dt`` cache of each layer.
    dt_cache_slot_strides : torch.Tensor
        One-dimensional CUDA int64 table of shape ``(L,)`` giving the ``dt``
        cache outer-slot stride, in elements, for each layer.
    A_ptrs : torch.Tensor
        One-dimensional CUDA int64 pointer table of shape ``(L,)`` for each
        layer's ``A`` vector.
    state_scale_ptrs : torch.Tensor
        One-dimensional CUDA int64 pointer table of shape ``(L,)`` for
        quantized-state scales. Use zero entries for non-quantized state.
    state_scale_slot_strides : torch.Tensor
        One-dimensional CUDA int64 table of shape ``(L,)`` giving the scale
        outer-slot stride, in elements, for each layer.
    src_slots, dst_slots : torch.Tensor
        CUDA int32 tensors of shape ``(L, B)`` selecting source and destination
        slots for each layer and physical batch request. A request/layer is
        skipped when either slot equals ``pad_slot_id``.
    ring_start, replay_prefix_len : torch.Tensor
        CUDA int32 tensors of shape ``(B,)``. ``replay_prefix_len[b]`` is the
        number of ring entries, beginning at ``ring_start[b]``, to apply to
        ``src_slots[:, b]`` before writing ``dst_slots[:, b]``. Zero performs
        an exact state/scale copy. For a shared source-state/ring snapshot, it
        must not exceed checkpointing SSU's ``prev_num_accepted_tokens`` for
        that request; this operation does not receive or validate that tracker.
    active_request_indices : torch.Tensor
        CUDA int32 tensor of shape ``(B,)``. Its prefix contains the indices in the batch (B dim above)
        for every request whose replay prefix length is non-negative, exactly
        once and in any order; remaining entries are ``-1``. The kernel stops
        at the first ``-1``. Tensor contents are not checked for consistency
        with ``replay_prefix_len``.
    state_dtype : torch.dtype
        JIT state-storage dtype. One-byte state requires ``dim=64`` and
        ``dstate=128``.
    input_dtype : torch.dtype
        JIT dtype of ``x`` and ``B`` cache entries.
    matrixA_dtype : torch.dtype
        JIT dtype of the per-head ``A`` values.
    dim : int
        JIT state head dimension.
    dstate : int
        JIT SSM state dimension.
    num_heads : int
        Runtime number of heads in every layer.
    heads_per_group : int
        JIT number of heads sharing each ``B`` cache group; it must divide
        ``num_heads``.
    max_window : int
        JIT maximum replay prefix length, in ring entries, in ``[1, 16]``.
    ring_buffer_len : int
        Runtime number of rows in every replay ring.
    pad_slot_id : int
        Slot sentinel to skip, matching :func:`flashinfer.mamba.checkpointing_ssu`.
        Defaults to ``-1``; zero is a valid slot unless passed here explicitly.
    rand_seed : Optional[torch.Tensor]
        One-element CUDA int64 seed required when ``philox_rounds > 0``.
    philox_rounds : int
        JIT number of Philox stochastic-rounding rounds. Zero disables
        stochastic rounding.

    Notes
    -----
    Source state and replay rings are read-only. Caller-owned slot tables and
    metadata select the tokens to apply; this operation has no acceptance,
    alignment, or cursor-update semantics.

    For example, if ``src_slots`` selects the state after token 123 and seven
    subsequent tokens have been accepted into the ring, a prefix cache that
    materializes states every 128 tokens passes ``replay_prefix_len=5`` to
    write the state after token 128.

    """
    tables = (
        state_ptrs,
        state_slot_strides,
        x_cache_ptrs,
        x_cache_slot_strides,
        B_cache_ptrs,
        B_cache_slot_strides,
        dt_cache_ptrs,
        dt_cache_slot_strides,
        A_ptrs,
        state_scale_ptrs,
        state_scale_slot_strides,
    )
    if any(t.dtype != torch.int64 or t.dim() != 1 or not t.is_cuda for t in tables):
        raise ValueError("all pointer/stride tables must be 1D CUDA int64 tensors")
    if (
        not src_slots.is_cuda
        or not dst_slots.is_cuda
        or src_slots.dtype != torch.int32
        or dst_slots.dtype != torch.int32
    ):
        raise ValueError("src_slots and dst_slots must be CUDA int32")
    if (
        not ring_start.is_cuda
        or not replay_prefix_len.is_cuda
        or not active_request_indices.is_cuda
        or ring_start.dtype != torch.int32
        or replay_prefix_len.dtype != torch.int32
        or active_request_indices.dtype != torch.int32
    ):
        raise ValueError("ring metadata and active_request_indices must be CUDA int32")
    if max_window < 1 or max_window > 16:
        raise ValueError("max_window must be in [1, 16]")
    if ring_buffer_len < 1:
        raise ValueError("ring_buffer_len must be positive")
    if num_heads < 1:
        raise ValueError("num_heads must be positive")
    if heads_per_group < 1 or num_heads % heads_per_group:
        raise ValueError("heads_per_group must be a positive divisor of num_heads")
    if state_dtype in (torch.int8, torch.float8_e4m3fn) and (dim, dstate) != (64, 128):
        raise ValueError(
            "8-bit state materialization currently requires dim=64 and dstate=128"
        )
    layers, batch = src_slots.shape
    if layers < 1:
        raise ValueError("src_slots must have at least one layer")
    if (
        dst_slots.shape != (layers, batch)
        or ring_start.numel() != batch
        or replay_prefix_len.numel() != batch
        or active_request_indices.numel() != batch
    ):
        raise ValueError("slot and metadata shapes are inconsistent")
    _module(
        state_dtype,
        input_dtype,
        matrixA_dtype,
        dim,
        dstate,
        heads_per_group,
        max_window,
        philox_rounds,
    ).replayssm_materialize(
        *tables,
        src_slots,
        dst_slots,
        ring_start,
        replay_prefix_len,
        active_request_indices,
        layers,
        num_heads,
        ring_buffer_len,
        pad_slot_id,
        rand_seed,
    )
