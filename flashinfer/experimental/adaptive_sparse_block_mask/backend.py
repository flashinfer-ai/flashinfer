"""Validation and launch adapter for adaptive sparse block-mask selection."""

from __future__ import annotations

import math
import os

import torch

from .jit import load_adaptive_sparse_block_mask_module


def _check_vector(
    value: torch.Tensor, name: str, batch_size: int, device: torch.device
) -> None:
    if value.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {value.dtype}")
    if tuple(value.shape) != (batch_size,):
        raise ValueError(f"{name} must have shape ({batch_size},)")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_length_values(
    q_seq_lens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    num_prompt_tokens: torch.Tensor,
    *,
    max_q_blocks: int,
    max_k_blocks: int,
    block_size: int,
) -> None:
    """Validate CUDA-resident lengths with one device-to-host transfer."""

    q_values, kv_values, prompt_values = torch.stack(
        (q_seq_lens, kv_seq_lens, num_prompt_tokens)
    ).tolist()
    max_q_tokens = int(max_q_blocks) * int(block_size)
    max_k_tokens = int(max_k_blocks) * int(block_size)

    for name, values in (
        ("q_seq_lens", q_values),
        ("kv_seq_lens", kv_values),
        ("num_prompt_tokens", prompt_values),
    ):
        if any(value < 0 for value in values):
            raise ValueError(f"{name} must contain nonnegative lengths")

    if any(value > max_q_tokens for value in q_values):
        raise ValueError(
            f"q_seq_lens must not exceed max_q_blocks * block_size ({max_q_tokens})"
        )
    if any(value > max_k_tokens for value in kv_values):
        raise ValueError(
            f"kv_seq_lens must not exceed max_k_blocks * block_size ({max_k_tokens})"
        )

    # Python integers are unbounded, so this calculation also verifies the
    # largest prompt value without reproducing the kernel's former int32 add.
    max_prompt_blocks = max(
        (
            (int(value) + int(block_size) - 1) // int(block_size)
            for value in prompt_values
        ),
        default=0,
    )
    if max_prompt_blocks > torch.iinfo(torch.int32).max:
        raise ValueError("num_prompt_tokens produces too many prompt blocks")


def _sync_input_validation_enabled() -> bool:
    """Return whether CUDA-resident value checks should synchronize to host."""

    return os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("0", "")


def run(
    block_logits: torch.Tensor,
    q_seq_lens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    num_prompt_tokens: torch.Tensor,
    *,
    block_size: int = 128,
    alpha: float = 1.0,
    initial_blocks: int = 4,
    window_size: int = 4,
    medium_rate: float = 0.2,
    medium_bias: int = 30,
    large_rate: float = 0.1,
    large_bias: int = 30,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if block_logits.ndim != 4:
        raise ValueError("block_logits must have shape [B, H, Qb, Kb]")
    if block_logits.dtype != torch.bfloat16:
        raise TypeError("block_logits must have dtype torch.bfloat16")
    if not block_logits.is_cuda:
        raise ValueError("block_logits must be a CUDA tensor")
    if not block_logits.is_contiguous():
        raise ValueError("block_logits must be contiguous")

    batch_size, num_heads, max_q_blocks, max_k_blocks = block_logits.shape
    if num_heads <= 0 or max_k_blocks <= 0:
        raise ValueError("num_heads and max_k_blocks must be positive")
    if max_k_blocks > 32768:
        raise ValueError("max_k_blocks must not exceed 32768")
    for value, name in (
        (q_seq_lens, "q_seq_lens"),
        (kv_seq_lens, "kv_seq_lens"),
        (num_prompt_tokens, "num_prompt_tokens"),
    ):
        _check_vector(value, name, batch_size, block_logits.device)

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_size > torch.iinfo(torch.int32).max:
        raise ValueError("block_size must fit in int32")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if initial_blocks < 0 or window_size < 0:
        raise ValueError("initial_blocks and window_size must be nonnegative")
    for value, name in ((medium_rate, "medium_rate"), (large_rate, "large_rate")):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if medium_bias < 0 or large_bias < 0:
        raise ValueError("medium_bias and large_bias must be nonnegative")

    if _sync_input_validation_enabled():
        _check_length_values(
            q_seq_lens,
            kv_seq_lens,
            num_prompt_tokens,
            max_q_blocks=max_q_blocks,
            max_k_blocks=max_k_blocks,
            block_size=block_size,
        )

    if out is None:
        out = torch.zeros_like(block_logits, dtype=torch.bool)
    else:
        if out.dtype != torch.bool:
            raise TypeError("out must have dtype torch.bool")
        if out.shape != block_logits.shape:
            raise ValueError("out must have the same shape as block_logits")
        if out.device != block_logits.device:
            raise ValueError("out must be on the same device as block_logits")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.data_ptr() == block_logits.data_ptr():
            raise ValueError("out must not alias block_logits")
        out.zero_()

    if batch_size == 0 or max_q_blocks == 0:
        return out

    module = load_adaptive_sparse_block_mask_module()
    with torch.cuda.device(block_logits.device):
        module.adaptive_sparse_block_mask(
            block_logits,
            q_seq_lens,
            kv_seq_lens,
            num_prompt_tokens,
            out,
            block_size,
            alpha,
            initial_blocks,
            window_size,
            medium_rate,
            medium_bias,
            large_rate,
            large_bias,
        )
    return out


__all__ = ["run"]
