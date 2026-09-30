"""Triton kernels and opaque prepared dispatch for BF16 vocabulary projection."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from b12x.preparation.types import plan_from_handle, require_prepared


@triton.jit
def _row_kernel(
    source,
    weight,
    output,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N: tl.constexpr,
):
    vocab_row = tl.program_id(0)
    token_row = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    values = tl.load(source + token_row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(
        weight + vocab_row * K + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output + token_row * N + vocab_row, tl.sum(values * weights, axis=0))


@triton.jit
def _row_loop_kernel(
    source,
    weight,
    output,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N: tl.constexpr,
):
    vocab_row = tl.program_id(0)
    token_row = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((), tl.float32)
    for start in range(0, K, BLOCK_K):
        positions = start + offsets
        mask = positions < K
        values = tl.load(source + token_row * K + positions, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(
            weight + vocab_row * K + positions,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(values * weights, axis=0)
    tl.store(output + token_row * N + vocab_row, accumulator)


@torch.library.custom_op("b12x::bf16_vocab_projection", mutates_args=())
def bf16_vocab_projection(
    source: torch.Tensor,
    weight: torch.Tensor,
    plan_handle: int,
) -> torch.Tensor:
    """Execute an already prepared Torch or Triton projection backend."""
    state = require_prepared(plan_from_handle(plan_handle), "gemm.bf16_vocab_projection", source.device)
    return state.run(source, weight)


@bf16_vocab_projection.register_fake
def _bf16_vocab_projection_fake(
    source: torch.Tensor,
    weight: torch.Tensor,
    plan_handle: int,
) -> torch.Tensor:
    del plan_handle
    return source.new_empty((source.shape[0], weight.shape[0]))


__all__ = ["bf16_vocab_projection"]
