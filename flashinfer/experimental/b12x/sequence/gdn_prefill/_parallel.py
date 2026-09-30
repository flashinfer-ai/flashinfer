"""Planned workspace and bindings for segment-parallel GDN recurrence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from b12x._lib.scratch import scratch_buffer_spec
from b12x._lib.scratch_layout import SCRATCH_ALIGN_BYTES, align_up, materialize_scratch_view

if TYPE_CHECKING:
    from ._impl import Binding, _Layout


@dataclass(frozen=True)
class ParallelPlan:
    """All segment summaries and incoming states are capacity-sized scratch."""

    segment_plan: _Layout
    segment_tokens: int
    max_segments: int
    reuse_outputs: bool
    regions: dict[str, tuple[int, tuple[int, ...], torch.dtype]]


@dataclass(frozen=True)
class ParallelBinding:
    _state: ParallelPlan
    transfer: Binding
    local_state: Binding
    output: Binding
    seq_segments: torch.Tensor
    pool: torch.Tensor
    packed_transfer: torch.Tensor
    transfer_flags: torch.Tensor


def materialize(plan: _Layout, *, segment_tokens: int) -> _Layout:
    from .._shared.delta_prefill.contract import materialize_layout

    caps = plan.caps
    segments = (caps.max_tokens + segment_tokens - 1) // segment_tokens + caps.max_seqs
    reuse_outputs = plan.k_split == 1
    slots = 2 + (4 if reuse_outputs else 3) * segments + caps.max_seqs
    inner_caps = replace(caps, max_seqs=segments, max_state_slots=slots, null_state_index=0)
    inner = materialize_layout(
        inner_caps, layout_type=type(plan), v_split=plan.v_split, k_split=plan.k_split, stages=plan.stages,
        window_tiles=inner_caps.tiles_capacity, workspace_windows=1,
        max_sequence_tiles=segment_tokens // 16,
    )
    regions = {}
    cursor = plan.scratch_specs()[0].shape[0]
    for name, shape, dtype in (
        ("segment_scratch", inner.scratch_specs()[0].shape, torch.uint8),
        ("pool", (slots, caps.heads, 128, 128), torch.float32),
        ("packed_transfer", (segments, caps.heads, 128, 128), torch.bfloat16),
        ("transfer_flags", (segments, caps.heads), torch.int32),
        ("seq_segments", (caps.max_seqs + 1,), torch.int32),
        ("cu_seqlens", (segments + 1,), torch.int32),
        ("zero_indices", (segments,), torch.int32),
        ("identity_indices", (segments,), torch.int32),
        ("transfer_indices", (segments,), torch.int32),
        ("local_indices", (segments,), torch.int32),
        ("output_indices", (segments,), torch.int32),
        ("checkpoint_indices", (segments,), torch.int32),
        ("checkpoint_offsets", (segments,), torch.int32),
        ("no_checkpoints", (segments,), torch.int32),
        ("num_seqs", (1,), torch.int32),
        ("num_tokens", (1,), torch.int32),
    ):
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        regions[name] = (cursor, shape, dtype)
        elements = 1
        for dimension in shape:
            elements *= dimension
        cursor += elements * dtype.itemsize
    parallel = ParallelPlan(inner, segment_tokens, segments, reuse_outputs, regions)
    return replace(plan, parallel=parallel, _scratch_specs=(
        scratch_buffer_spec(caps.op_name, nbytes=cursor, device=caps.device),
    ))


def bind(binding: Binding) -> ParallelBinding:
    from ._impl import _bind as bind_segment

    plan = binding._state.parallel
    assert plan is not None
    views = {name: materialize_scratch_view(binding.scratch, offset_bytes=offset,
                                           shape=shape, dtype=dtype)[0]
             for name, (offset, shape, dtype) in plan.regions.items()}
    shared = dict(
        scratch=views["segment_scratch"], q=binding.q, k=binding.k, v=binding.v,
        a=binding.a, b=binding.b, A_log=binding.A_log, dt_bias=binding.dt_bias,
        recurrent_state=views["pool"], cu_seqlens=views["cu_seqlens"],
        num_seqs=views["num_seqs"], num_tokens=views["num_tokens"], output=binding.output,
    )
    transfer = bind_segment(
        plan.segment_plan, **shared, initial_state_indices=views["identity_indices"],
        final_state_indices=views["transfer_indices"],
        checkpoint_state_indices=views["no_checkpoints"], checkpoint_offsets=views["no_checkpoints"],
    )
    local = bind_segment(
        plan.segment_plan, **shared, initial_state_indices=views["zero_indices"],
        final_state_indices=views["local_indices"],
        checkpoint_state_indices=views["checkpoint_indices"], checkpoint_offsets=views["checkpoint_offsets"],
    )
    output = bind_segment(
        plan.segment_plan, **shared, initial_state_indices=views["output_indices"],
        final_state_indices=views["output_indices"],
        checkpoint_state_indices=views["checkpoint_indices"], checkpoint_offsets=views["checkpoint_offsets"],
    )
    return ParallelBinding(plan, transfer, local, output, views["seq_segments"], views["pool"],
                           views["packed_transfer"], views["transfer_flags"])


def compile_binding(binding: Binding) -> tuple:
    from .._shared.delta_prefill._cute_kernels import (
        _compile_prepare, _compile_prologue, _compile_recurrence,
    )
    from ._parallel_kernels import compile_auxiliary

    parallel = binding.parallel
    assert parallel is not None
    return (
        _compile_prologue(binding)[1],
        _compile_prologue(parallel.output)[1],
        _compile_prepare(parallel.output)[1],
        _compile_recurrence(parallel.transfer, 2)[1],
        _compile_recurrence(parallel.local_state, 4 if parallel._state.reuse_outputs else 1)[1],
        _compile_recurrence(parallel.output, 5 if parallel._state.reuse_outputs else 0)[1],
        compile_auxiliary(binding),
    )




def run(binding: Binding, *, programs: tuple, scale: float, eps: float) -> None:
    parallel = binding.parallel
    if parallel is None:
        raise TypeError("parallel prefill requires a segment binding")
    prologue, segment_prologue, prepare, transfer, local, output, auxiliary = programs
    prologue(binding)
    auxiliary.partition(binding)
    segment_prologue(parallel.output)
    prepare(parallel.output, scale, 0.0, eps, 0)
    transfer(parallel.transfer, 0)
    local(parallel.local_state, 0)
    auxiliary.pack_transfer(binding)
    auxiliary.boundaries(binding)
    output(parallel.output, 0)
    auxiliary.commit(binding)
