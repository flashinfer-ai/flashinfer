"""Metadata stages for draft-round selection reuse inside the QSA operation."""

from __future__ import annotations

import torch
from b12x._lib.compile_plan import launch_triton as _launch_triton
import triton
import triton.language as tl


@torch.library.custom_op("b12x::qsa_validate_draft_buffers", mutates_args=("mutable",))
def validate_buffers(mutable: list[torch.Tensor], inputs: list[torch.Tensor]) -> None:
    """Check addresses before writes, outside Dynamo's symbolic tracing.

    The mutation annotation orders validation before consumers of these buffers.
    The check launches no GPU work and does not change buffer contents.
    """
    from ._contract import _require_mutation_alias_contract

    _require_mutation_alias_contract(
        mutable=tuple((f"draft buffer {i}", t) for i, t in enumerate(mutable)),
        read_only=tuple((f"draft input {i}", t) for i, t in enumerate(inputs)),
    )


@validate_buffers.register_fake
def _validate_fake(mutable: list[torch.Tensor], inputs: list[torch.Tensor]) -> None:
    return None


@triton.jit(do_not_specialize=["rows", "enabled"])
def _record_kernel(
    positions,
    saved_positions,
    saved_rows,
    selection,
    saved_selection,
    rows,
    enabled,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK)
    active = (row < rows) & (enabled != 0)
    tl.store(saved_positions + row, tl.load(positions + row, active, other=-1), active)
    offset = row * tl.full((), WIDTH, tl.int64) + column
    value = tl.load(selection + offset, active & (column < WIDTH), other=-1)
    tl.store(saved_selection + offset, value, active & (column < WIDTH))
    if tl.program_id(0) == 0:
        tl.store(saved_rows, rows, enabled != 0)


@torch.library.custom_op("b12x::qsa_reset_draft_anchors", mutates_args=("storage",))
def reset_anchors(storage: torch.Tensor, count_offset: int) -> None:
    from ._contract import _scratch_view

    _scratch_view(
        storage, offset_bytes=count_offset, shape=(1,), dtype=torch.int32
    ).zero_()


@reset_anchors.register_fake
def _reset_fake(storage: torch.Tensor, count_offset: int) -> None:
    return None


@torch.library.custom_op(
    "b12x::qsa_record_draft_anchors",
    mutates_args=("storage",),
)
def record_anchors(
    positions: torch.Tensor,
    selection: torch.Tensor,
    storage: torch.Tensor,
    source_capacity: int,
    width: int,
    enabled: bool,
) -> None:
    from ._contract import DraftSelectionPlan

    state = DraftSelectionPlan(storage.device, source_capacity, width).bind(
        storage=storage
    )
    rows = int(positions.shape[0])
    _launch_triton(
        _record_kernel,
        (rows,),
        positions,
        state.logical_positions,
        state.num_source_rows,
        selection,
        state.selected_positions,
        rows,
        int(enabled),
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
    )


@record_anchors.register_fake
def _record_fake(
    positions: torch.Tensor,
    selection: torch.Tensor,
    storage: torch.Tensor,
    source_capacity: int,
    width: int,
    enabled: bool,
) -> None:
    return None


@triton.jit(do_not_specialize=["rows", "source_capacity", "max_requests"])
def _prepare_kernel(
    source_positions,
    source_selection,
    source_rows,
    num_source_rows,
    request_ids,
    query_positions,
    selected,
    rows,
    source_capacity,
    max_requests,
    WIDTH: tl.constexpr,
    TAIL: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK)
    request = tl.load(request_ids + row, row < rows, other=-1).to(tl.int64)
    mapped = (request >= 0) & (request < max_requests)
    source = tl.load(source_rows + request, mapped, other=-1).to(tl.int64)
    source_valid = (
        mapped
        & (source >= 0)
        & (source < source_capacity)
        & (source < tl.load(num_source_rows))
    )
    anchor = tl.load(source_positions + source, source_valid, other=-1)
    position = tl.load(query_positions + row, row < rows, other=-1)
    start = anchor + 1
    valid = (
        source_valid & (anchor >= 0) & (position >= start) & (position < start + TAIL)
    )
    original = tl.load(
        source_selection + source * tl.full((), WIDTH, tl.int64) + column,
        valid & (column < WIDTH),
        other=-1,
    )
    tail = start + column - WIDTH
    if DCP_SIZE == 1:
        value = tl.where(
            column < WIDTH,
            original,
            tl.where(valid & (tail <= position), tail, -1),
        )
    else:
        stripe = tail // CP_INTERLEAVE
        owner = stripe % DCP_SIZE
        local_tail = (
            (stripe // DCP_SIZE) * CP_INTERLEAVE
            + tail % CP_INTERLEAVE
        )
        value = tl.where(
            column < WIDTH,
            original,
            tl.where(
                valid & (tail <= position) & (owner == DCP_RANK),
                local_tail,
                -1,
            ),
        )
    tl.store(
        selected + row * tl.full((), WIDTH + TAIL, tl.int64) + column,
        value,
        column < WIDTH + TAIL,
    )


@torch.library.custom_op("b12x::qsa_prepare_draft_selection", mutates_args=("scratch",))
def prepare_selection(
    storage: torch.Tensor,
    source_capacity: int,
    width: int,
    source_rows: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    scratch: torch.Tensor,
    selected_offset: int,
    max_requests: int,
    tail: int,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> None:
    from ._contract import DraftSelectionPlan, _scratch_view

    state = DraftSelectionPlan(storage.device, source_capacity, width).bind(
        storage=storage
    )
    selected = _scratch_view(
        scratch,
        offset_bytes=selected_offset,
        shape=(max_requests, width + tail),
        dtype=torch.int32,
    )
    rows = int(query_positions.shape[0])
    _launch_triton(
        _prepare_kernel,
        (rows,),
        state.logical_positions,
        state.selected_positions,
        source_rows,
        state.num_source_rows,
        request_ids,
        query_positions,
        selected,
        rows,
        source_capacity,
        max_requests,
        WIDTH=width,
        TAIL=tail,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        CP_INTERLEAVE=cp_kv_cache_interleave_size,
        BLOCK=triton.next_power_of_2(width + tail),
        num_warps=4,
    )


@prepare_selection.register_fake
def _prepare_fake(
    storage: torch.Tensor,
    source_capacity: int,
    width: int,
    source_rows: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    scratch: torch.Tensor,
    selected_offset: int,
    max_requests: int,
    tail: int,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> None:
    return None
