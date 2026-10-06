"""Lightweight dispatch contract for the MTP CuTe projection GEMMs."""

from __future__ import annotations

import torch


TMA_ALIGNMENT_BYTES = 16
PROJECTION_ROW_ALIGNMENT = 16
QWEN_STREAMS = 4
QWEN_HIDDEN_SIZE = 2_560


def _padded_rows(rows: int) -> int:
    return (
        (int(rows) + PROJECTION_ROW_ALIGNMENT - 1) // PROJECTION_ROW_ALIGNMENT
    ) * PROJECTION_ROW_ALIGNMENT


def is_qwen_projection_contract(*, streams: int, hidden_size: int) -> bool:
    """Return whether the logical shape is the Qwen3.8 MTP contract."""
    return int(streams) == QWEN_STREAMS and int(hidden_size) == QWEN_HIDDEN_SIZE


def supports_prefill(
    *,
    tokens: int,
    streams: int,
    hidden_size: int,
) -> bool:
    """Return whether the mandatory Qwen3.8 CuTe projection path applies."""
    token_count = int(tokens)
    if token_count <= 0 or not is_qwen_projection_contract(
        streams=streams,
        hidden_size=hidden_size,
    ):
        return False
    return True


def projection_capacity_rows(
    *,
    max_tokens: int,
    streams: int,
    hidden_size: int,
) -> tuple[int, int]:
    """Return capacity-shaped rows for the two mandatory CuTe projections."""
    capacity = int(max_tokens)
    stream_count = int(streams)
    if capacity <= 0:
        raise ValueError(f"max_tokens must be positive, got {capacity}")
    if not is_qwen_projection_contract(
        streams=stream_count,
        hidden_size=hidden_size,
    ):
        raise ValueError(
            "MTP feedback only implements the Qwen3.8 CuTe projection "
            f"contract S={QWEN_STREAMS},H={QWEN_HIDDEN_SIZE}; got "
            f"S={stream_count},H={int(hidden_size)}"
        )

    return _padded_rows(capacity), _padded_rows(capacity * stream_count)


def tensors_support_prefill(*tensors: torch.Tensor) -> bool:
    """Return whether tensors satisfy the contiguous TMA pointer contract."""
    return all(
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.is_contiguous()
        and int(tensor.data_ptr()) % TMA_ALIGNMENT_BYTES == 0
        for tensor in tensors
    )


def require_qwen_cute_tensors(**tensors: torch.Tensor) -> None:
    """Validate the contiguous TMA tensor contract for Qwen projections."""
    invalid = [
        name
        for name, tensor in tensors.items()
        if not tensors_support_prefill(tensor)
    ]
    if invalid:
        raise ValueError(
            "Qwen MTP CuTe projection contract violation: "
            + ", ".join(invalid)
            + " must be contiguous CUDA tensors with 16-byte-aligned TMA "
            "pointers"
        )


__all__ = [
    "PROJECTION_ROW_ALIGNMENT",
    "QWEN_HIDDEN_SIZE",
    "QWEN_STREAMS",
    "TMA_ALIGNMENT_BYTES",
    "is_qwen_projection_contract",
    "projection_capacity_rows",
    "require_qwen_cute_tensors",
    "supports_prefill",
    "tensors_support_prefill",
]
