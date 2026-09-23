# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Production K3 reduction for the opt-in owner-local BF16 combine path.

K2 writes one BF16 partial per source token and expert-owner rank. A
graph-ordered one-CTA finalizer supplies the cross-rank visibility boundary,
then this vector K3 widens selected owner partials to FP32, sums them in
ascending owner order, and returns BF16. This intentionally adds one BF16
rounding point relative to the ordinary top-k-slot reduction.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cutlass_dsl import Float32, Int32

from .moe_utils import spin_wait


REDUCE_THREADS = 512
REDUCE_HIDDEN_PER_THREAD = 8


def _to_cute_tensor(
    tensor: torch.Tensor, *, assumed_align: int = 16
) -> cute.Tensor:
    result = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    return result.mark_layout_dynamic(
        leading_dim=cutlass_torch.get_leading_dim(tensor)
    )


@cute.kernel
def rank_partial_reduce_bf16_kernel(
    combine_output: cute.Tensor,
    ready_flags: cute.Tensor,
    reduced_output: cute.Tensor,
    topk_idx: cute.Tensor,
    world_size: cutlass.Constexpr[int],
    num_topk: cutlass.Constexpr[int],
    num_experts_per_rank: cutlass.Constexpr[int],
    tokens: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    hidden_blocks: cutlass.Constexpr[int],
):
    linear_block_idx, _, _ = cute.arch.block_idx()
    hidden_block = linear_block_idx % Int32(hidden_blocks)
    token_idx = linear_block_idx // Int32(hidden_blocks)
    tid = cute.arch.thread_idx()[0]
    block_dim = cute.arch.block_dim()[0]

    rank_mask = Int32(0)
    for slot in cutlass.range_constexpr(num_topk):
        expert = Int32(topk_idx[token_idx, Int32(slot)])
        if expert >= Int32(0):
            rank_mask |= Int32(1) << (
                expert // Int32(num_experts_per_rank)
            )

    expected_epoch = ready_flags[Int32(tokens), Int32(0)]
    if tid == Int32(0):
        for rank in cutlass.range_constexpr(world_size):
            if rank_mask & Int32(1 << rank):
                spin_wait(
                    ready_flags.iterator
                    + token_idx * Int32(world_size)
                    + Int32(rank),
                    lambda value: value == expected_epoch,
                    fail_sleep_cycles=200,
                )
    cute.arch.sync_threads()

    hidden_base = (
        hidden_block * block_dim * Int32(REDUCE_HIDDEN_PER_THREAD) + tid
    )
    if hidden_base < Int32(hidden):
        for elem in cutlass.range_constexpr(REDUCE_HIDDEN_PER_THREAD):
            hidden_idx = hidden_base + Int32(elem) * block_dim
            if hidden_idx < Int32(hidden):
                value = Float32(0.0)
                for rank in cutlass.range_constexpr(world_size):
                    if rank_mask & Int32(1 << rank):
                        value += Float32(
                            combine_output[
                                token_idx, Int32(rank), hidden_idx
                            ]
                        )
                reduced_output[token_idx, hidden_idx] = value.to(
                    cutlass.BFloat16
                )


def compile_rank_local_combine(
    combine_output: torch.Tensor,
    ready_flags: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    world_size: int,
    num_experts_per_rank: int,
    stream: cuda.CUstream,
) -> tuple[Any, dict[str, Any]]:
    """Compile the owner-partial K3 reduction for one fixed problem shape."""

    if combine_output.dtype != torch.bfloat16 or combine_output.ndim != 3:
        raise TypeError("rank-local combine_output must be a 3-D BF16 tensor")
    if ready_flags.dtype != torch.int32 or ready_flags.ndim != 2:
        raise TypeError("rank-local ready_flags must be a 2-D Int32 tensor")
    if reduced_output.dtype != torch.bfloat16 or reduced_output.ndim != 2:
        raise TypeError("rank-local reduced_output must be a 2-D BF16 tensor")

    tokens, hidden = map(int, reduced_output.shape)
    num_topk = int(combine_output.shape[1])
    if tuple(combine_output.shape) != (tokens, num_topk, hidden):
        raise ValueError("rank-local combine_output shape does not match output")
    if tuple(ready_flags.shape) != (tokens + 1, world_size):
        raise ValueError("rank-local ready_flags shape does not match EP geometry")
    if tuple(topk_ids.shape) != (tokens, num_topk):
        raise ValueError("rank-local topk_ids shape does not match EP geometry")
    if not 0 < world_size <= num_topk or num_experts_per_rank <= 0:
        raise ValueError("invalid rank-local owner geometry")

    combine_cute = _to_cute_tensor(combine_output)
    ready_cute = _to_cute_tensor(ready_flags, assumed_align=4)
    reduced_cute = _to_cute_tensor(reduced_output)
    topk_cute = _to_cute_tensor(topk_ids)
    hidden_blocks = (
        hidden + REDUCE_THREADS * REDUCE_HIDDEN_PER_THREAD - 1
    ) // (REDUCE_THREADS * REDUCE_HIDDEN_PER_THREAD)

    @cute.jit
    def _launcher(
        combine_cute: cute.Tensor,
        ready_cute: cute.Tensor,
        reduced_cute: cute.Tensor,
        topk_cute: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rank_partial_reduce_bf16_kernel(
            combine_cute,
            ready_cute,
            reduced_cute,
            topk_cute,
            world_size=world_size,
            num_topk=num_topk,
            num_experts_per_rank=num_experts_per_rank,
            tokens=tokens,
            hidden=hidden,
            hidden_blocks=hidden_blocks,
        ).launch(
            grid=[tokens * hidden_blocks, 1, 1],
            block=[REDUCE_THREADS, 1, 1],
            stream=stream,
        )

    compiled = cute.compile(
        _launcher,
        combine_cute,
        ready_cute,
        reduced_cute,
        topk_cute,
        stream,
    )
    runtime = dict(
        combine_cute=combine_cute,
        ready_cute=ready_cute,
        reduced_cute=reduced_cute,
        topk_cute=topk_cute,
        stream=stream,
    )
    return compiled, runtime


__all__ = ["compile_rank_local_combine", "rank_partial_reduce_bf16_kernel"]
