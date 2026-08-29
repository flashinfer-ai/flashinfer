"""Materialize immutable ReplaySSM prefix states from replay rings."""

import functools
from typing import Optional
import torch
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
    flush_count: torch.Tensor,
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
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 0,
) -> None:
    """Materialize selected ReplaySSM states without touching source cache state.

    Pointer and stride tables are CUDA int64 tensors indexed by layer. Slots are
    CUDA int32 tensors with shape ``[num_layers, batch]``; ring metadata is
    non-layer ``[batch]``. ``flush_count < 0`` is a no-op and zero is an exact
    raw state/scale copy.
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
        or not flush_count.is_cuda
        or ring_start.dtype != torch.int32
        or flush_count.dtype != torch.int32
    ):
        raise ValueError("ring_start and flush_count must be CUDA int32")
    if max_window < 1 or max_window > 16:
        raise ValueError("max_window must be in [1, 16]")
    if ring_buffer_len < 1:
        raise ValueError("ring_buffer_len must be positive")
    if num_heads < 1:
        raise ValueError("num_heads must be positive")
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
        or flush_count.numel() != batch
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
        flush_count,
        layers,
        num_heads,
        ring_buffer_len,
        rand_seed,
    )
