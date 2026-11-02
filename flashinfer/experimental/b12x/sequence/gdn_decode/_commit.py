"""Caller-owned binding for grouped KDA accepted-state recovery."""

from dataclasses import dataclass
import math

import torch

from b12x._lib.scratch import scratch_tensor
from b12x.preparation.types import require_prepared
from ._impl import _require_tensor


@dataclass(frozen=True)
class KdaCommitBinding:
    """Borrowed address tables, metadata and workspace for multiple layers.

    Address tables must reference live pools with the head layout of the decode
    plan. Their owner must keep the pools alive through graph replay. Different
    requests must not write the same non-null checkpoint, and a written slot
    must not be another request's source in the same call. Source/final/boundary
    slots within one request may alias. If boundary and final destinations
    alias, the final state takes precedence; retaining an earlier boundary
    requires a distinct destination. Counts and indices are trusted device
    metadata; no CPU synchronization is performed when binding.
    """

    plan: object
    scratch: torch.Tensor
    tensors: tuple[torch.Tensor, ...]
    batch: int
    layers: int


def bind_kda_commit(
    plan,
    *,
    scratch,
    state_base_addrs,
    state_block_strides,
    correction_cache_base_addrs,
    correction_cache_block_strides,
    kg_cache_base_addrs,
    kg_cache_block_strides,
    A_log,
    dt_bias,
    state_indices,
    commit_lens,
    final_state_indices,
    boundary_state_indices,
    boundary_recovery_lens,
):
    """Bind precomputed commit lengths and optional aligned boundary slots.

    ``commit_lens`` must be between zero and the planned speculative window.
    A null source/final slot or zero commit length leaves all state unchanged.
    Null boundary slots suppress checkpoint export. No allocations or writes
    occur here; the caller owns preparation of all metadata and address tables.
    Base-address tables contain raw byte addresses. Block-stride tables contain
    element counts in their respective pool dtypes, not byte strides.
    Source indices use the plan's index dtype. Destination indices and recovery
    lengths must be Int32; destination slot IDs must fit that signed range.
    Pool-offset multiplication and pointer arithmetic are Int64 regardless of
    the index storage dtype. Binding rejects other destination dtypes; it does
    not cast or truncate them.
    """
    state = require_prepared(plan, "attention.gdn")
    q, caps = state.query, state.layout.caps
    if not q.recover_speculative_state:
        raise ValueError("commit requires a speculative-state recovery plan")
    scratch = scratch_tensor(
        scratch, state.layout.scratch_specs(), owner="KDA recovery"
    )
    layers, batch = state_base_addrs.numel(), state_indices.numel()
    if layers == 0 or not 0 < batch <= caps.max_seqs:
        raise ValueError("commit requires nonempty layers and a batch within capacity")
    tables = (
        state_base_addrs,
        state_block_strides,
        correction_cache_base_addrs,
        correction_cache_block_strides,
        kg_cache_base_addrs,
        kg_cache_block_strides,
    )
    for name, table in zip(
        (
            "state_base_addrs",
            "state_block_strides",
            "correction_cache_base_addrs",
            "correction_cache_block_strides",
            "kg_cache_base_addrs",
            "kg_cache_block_strides",
        ),
        tables,
        strict=True,
    ):
        _require_tensor(
            name,
            table,
            shape=(layers,),
            device=caps.device,
            dtypes=(torch.int64,),
        )
    _require_tensor(
        "A_log",
        A_log,
        shape=(layers, q.value_heads),
        device=caps.device,
        dtypes=(getattr(torch, q.a_log_dtype),),
    )
    _require_tensor(
        "dt_bias",
        dt_bias,
        shape=(layers, q.value_heads, 128),
        device=caps.device,
        dtypes=(getattr(torch, q.dt_bias_dtype),),
    )
    _require_tensor(
        "state_indices",
        state_indices,
        shape=(batch,),
        device=caps.device,
        dtypes=(getattr(torch, q.state_indices_dtype),),
        contiguous=False,
    )
    if state_indices.stride(0) <= 0:
        raise ValueError("state indices must have positive stride")
    metadata = (
        commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
    )
    for name, tensor in zip(
        (
            "commit_lens",
            "final_state_indices",
            "boundary_state_indices",
            "boundary_recovery_lens",
        ),
        metadata,
        strict=True,
    ):
        _require_tensor(
            name,
            tensor,
            shape=(batch,),
            device=caps.device,
            dtypes=(torch.int32,),
        )
    return KdaCommitBinding(
        plan,
        scratch,
        (*tables, A_log, dt_bias, state_indices, *metadata),
        batch,
        layers,
    )


def run_kda_commit(binding: KdaCommitBinding, *, lower_bound: float = -5.0) -> None:
    """Recover accepted FP32 checkpoints without re-evaluating the model."""
    lower_bound = float(lower_bound)
    if not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("KDA lower bound must be finite and negative")
    state = require_prepared(binding.plan, "attention.gdn")
    state.commit(binding, lower_bound=lower_bound)
