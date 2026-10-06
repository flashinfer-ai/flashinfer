"""Dispatch contract for the HyperConnection CuTe combine+norm kernel."""

from __future__ import annotations

import torch


VECTOR_ALIGNMENT_BYTES = 16


def is_qwen_combine_norm_contract(*, streams: int, hidden_size: int) -> bool:
    """Return whether the logical shape is the Qwen3.8 production contract."""
    return int(streams) == 4 and int(hidden_size) == 2_560


def supports_combine_norm(
    *,
    streams: int,
    hidden_size: int,
) -> bool:
    """Return whether the Qwen contract has a production CuTe implementation."""
    if not is_qwen_combine_norm_contract(
        streams=streams,
        hidden_size=hidden_size,
    ):
        return False
    return True


def require_cute_combine_norm(
    *,
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    combined: torch.Tensor,
    normalized: torch.Tensor,
    streams: int,
    hidden_size: int,
) -> None:
    """Require the production CuTe contract without permitting a fallback."""
    if not is_qwen_combine_norm_contract(
        streams=streams,
        hidden_size=hidden_size,
    ):
        raise ValueError(
            "HyperConnection CuTe combine+norm requires streams=4 and hidden_size=2560"
        )
    if not isinstance(state, torch.Tensor) or not state.is_cuda:
        raise RuntimeError(
            "Qwen3.8 HyperConnection combine+norm requires the CuTe CUDA path"
        )
    tokens = int(state.shape[0])
    tensors = (
        state,
        block_output,
        injection_logits,
        next_norm_weight,
        combined,
        normalized,
    )
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise RuntimeError(
                "Qwen3.8 HyperConnection combine+norm requires CUDA tensors"
            )
        if tensor.device != state.device:
            raise ValueError(
                "Qwen3.8 HyperConnection combine+norm tensors must share a device"
            )
        if tensor.dtype != torch.bfloat16:
            raise ValueError(
                "Qwen3.8 HyperConnection combine+norm tensors must use BF16"
            )
        if tensor is injection_logits:
            if tensor.ndim != 2 or tensor.stride(1) != 1 or tensor.stride(0) < streams:
                raise ValueError(
                    "Qwen3.8 HyperConnection injection logits require unit column "
                    "stride and disjoint rows"
                )
        elif not tensor.is_contiguous():
            raise ValueError(
                "Qwen3.8 HyperConnection combine+norm tensors must be contiguous"
            )

    shapes = {
        "state": (state, (tokens, int(streams) * int(hidden_size))),
        "block_output": (block_output, (tokens, int(hidden_size))),
        "injection_logits": (injection_logits, (tokens, int(streams))),
        "next_norm_weight": (
            next_norm_weight,
            (int(streams) * int(hidden_size),),
        ),
        "combined": (combined, tuple(state.shape)),
        "normalized": (normalized, tuple(state.shape)),
    }
    for name, (tensor, expected) in shapes.items():
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"Qwen3.8 HyperConnection combine+norm {name} must have shape "
                f"{expected}, got {tuple(tensor.shape)}"
            )

    vector_tensors = (
        state,
        block_output,
        next_norm_weight,
        combined,
        normalized,
    )
    if not all(
        int(tensor.data_ptr()) % VECTOR_ALIGNMENT_BYTES == 0
        for tensor in vector_tensors
    ):
        raise ValueError(
            "Qwen3.8 HyperConnection CuTe combine+norm vector tensors must be "
            f"{VECTOR_ALIGNMENT_BYTES}-byte aligned"
        )


__all__ = [
    "VECTOR_ALIGNMENT_BYTES",
    "is_qwen_combine_norm_contract",
    "require_cute_combine_norm",
    "supports_combine_norm",
]
