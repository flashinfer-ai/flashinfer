"""Opaque MTP feedback launches with mandatory Qwen CuTe projections."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from b12x.preparation.types import plan_from_handle, require_prepared
from ._cute_prefill_config import require_qwen_cute_tensors


@triton.jit
def _token_norm_kernel(
    token_embedding,
    token_norm_weight,
    token_normalized,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN_SIZE
    offsets = token * HIDDEN_SIZE + cols.to(tl.int64)
    values = tl.load(token_embedding + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / HIDDEN_SIZE
    normalized = values * tl.rsqrt(variance + eps)
    weight = tl.load(token_norm_weight + cols, mask=mask, other=0.0).to(tl.float32)
    result = (normalized * (1.0 + weight)).to(tl.bfloat16)
    tl.store(token_normalized + offsets, result, mask=mask)


@triton.jit
def _state_partial_sum_kernel(
    multi_state,
    state_partial_sums,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN_SIZE
    offsets = row * HIDDEN_SIZE + cols.to(tl.int64)
    values = tl.load(multi_state + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(state_partial_sums + row, tl.sum(values * values, axis=0))


@triton.jit
def _state_norm_kernel(
    multi_state,
    state_partial_sums,
    state_norm_weight,
    state_normalized,
    eps,
    STREAMS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1).to(tl.int64)
    stream_offsets = tl.arange(0, BLOCK_S)
    sum_squares = tl.sum(
        tl.load(
            state_partial_sums + token * STREAMS + stream_offsets,
            mask=stream_offsets < STREAMS,
            other=0.0,
        ),
        axis=0,
    )
    inverse_rms = tl.rsqrt(sum_squares / (STREAMS * HIDDEN_SIZE) + eps)

    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN_SIZE
    row = token * STREAMS + stream
    offsets = row * HIDDEN_SIZE + cols.to(tl.int64)
    weight_offsets = stream * HIDDEN_SIZE + cols.to(tl.int64)
    values = tl.load(multi_state + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(state_norm_weight + weight_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    result = (values * inverse_rms * (1.0 + weight)).to(tl.bfloat16)
    tl.store(state_normalized + offsets, result, mask=mask)


def _scratch_view(
    scratch: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    nbytes = numel * int(dtype.itemsize)
    return scratch.narrow(0, int(offset_bytes), nbytes).view(dtype).view(shape)


def _capacity_matrix(tensor: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    flat = tensor.reshape(-1)
    required = int(rows) * int(columns)
    available = (
        int(flat.untyped_storage().nbytes())
        - int(flat.storage_offset()) * int(flat.element_size())
    ) // int(flat.element_size())
    if required > available:
        raise ValueError(
            f"padded CuTe view needs {required} elements, storage has {available}"
        )
    return flat.as_strided((int(rows), int(columns)), (int(columns), 1))


def _qwen_cute_projections(
    token_normalized: torch.Tensor, state_normalized: torch.Tensor,
    embedding_fc_weight: torch.Tensor, hidden_fc_weight: torch.Tensor,
    token_path: torch.Tensor, output: torch.Tensor, *,
    projections: tuple, tokens: int, token_rows: int, state_rows: int,
    streams: int, hidden_size: int,
) -> None:
    token_input = _capacity_matrix(token_normalized, token_rows, hidden_size)
    token_output = _capacity_matrix(token_path, token_rows, hidden_size)
    state_input = _capacity_matrix(state_normalized, state_rows, hidden_size)
    token_projection, state_projection = projections
    with torch.cuda.device(token_normalized.device):
        token_projection(token_input, embedding_fc_weight, token_output, live_rows=tokens)
        state_projection(
            state_input, hidden_fc_weight, output.reshape(-1),
            token_path=token_output, live_rows=tokens * streams,
        )


def _launch_mtp_feedback(
    token_embedding, multi_state, token_norm_weight, state_norm_weight,
    embedding_fc_weight, hidden_fc_weight, scratch, output, eps, state,
) -> None:
    tokens = int(token_embedding.shape[0])
    if tokens == 0:
        return
    layout, programs = state.layout, state.programs
    caps = layout.caps
    h, s = caps.hidden_size, caps.streams
    token_normalized = _scratch_view(
        scratch, offset_bytes=layout.token_normalized_offset_bytes,
        shape=(layout.token_projection_rows, h), dtype=torch.bfloat16,
    )[:tokens]
    state_partial_sums = _scratch_view(
        scratch, offset_bytes=layout.state_partial_sums_offset_bytes,
        shape=(caps.max_tokens, s), dtype=torch.float32,
    )[:tokens]
    state_normalized = _scratch_view(
        scratch, offset_bytes=layout.state_normalized_offset_bytes,
        shape=(layout.state_projection_rows // s, s, h), dtype=torch.bfloat16,
    )[:tokens]
    token_path = _scratch_view(
        scratch, offset_bytes=layout.token_path_offset_bytes,
        shape=(layout.token_projection_rows, h), dtype=torch.bfloat16,
    )[:tokens]
    require_qwen_cute_tensors(
        token_normalized=token_normalized, state_normalized=state_normalized,
        embedding_fc_weight=embedding_fc_weight, hidden_fc_weight=hidden_fc_weight,
        token_path=token_path, output=output,
    )
    programs["token_norm"][(tokens, 1, 1)](
        token_embedding, token_norm_weight, token_normalized, float(eps), h, layout.norm_block_h,
    )
    programs["partial"][(tokens * s, 1, 1)](
        multi_state, state_partial_sums, h, layout.norm_block_h,
    )
    programs["state_norm"][(tokens, s, 1)](
        multi_state, state_partial_sums, state_norm_weight, state_normalized,
        float(eps), s, h, layout.norm_block_s, layout.norm_block_h,
    )
    _qwen_cute_projections(
        token_normalized, state_normalized, embedding_fc_weight, hidden_fc_weight,
        token_path, output, projections=programs["projections"], tokens=tokens,
        token_rows=layout.token_projection_rows, state_rows=layout.state_projection_rows,
        streams=s, hidden_size=h,
    )


@torch.library.custom_op(
    "b12x::mtp_feedback",
    mutates_args=("scratch", "output"),
)
def _mtp_feedback_op(
    token_embedding: torch.Tensor, multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor, state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor, hidden_fc_weight: torch.Tensor,
    scratch: torch.Tensor, output: torch.Tensor, eps: float, plan_handle: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "sequence.mtp_feedback", token_embedding.device)
    state.run_tensors(
        token_embedding, multi_state, token_norm_weight, state_norm_weight,
        embedding_fc_weight, hidden_fc_weight, scratch, output, eps=eps,
    )


@_mtp_feedback_op.register_fake
def _mtp_feedback_fake(
    token_embedding: torch.Tensor, multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor, state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor, hidden_fc_weight: torch.Tensor,
    scratch: torch.Tensor, output: torch.Tensor, eps: float, plan_handle: int,
) -> None:
    del token_embedding, multi_state, token_norm_weight, state_norm_weight
    del embedding_fc_weight, hidden_fc_weight, scratch, output, eps, plan_handle


def run_mtp_feedback(*tensors, eps, plan):
    torch.ops.b12x.mtp_feedback(*tensors, float(eps), plan.handle)


__all__ = ["run_mtp_feedback"]
