# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""PDL-aware deterministic finalize for W4A16's permuted GEMM2 output."""

import torch
import triton
import triton.language as tl

_MAX_BLOCK_K = 8192


@triton.jit
def _moe_unpermute_bf16_pdl_kernel(
    permuted_input,
    output,
    expanded_idx_to_permuted_idx,
    topk_scales,
    HIDDEN_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_idx = tl.program_id(0)
    route_offsets = tl.arange(0, ROUTE_BLOCK)
    route_mask = route_offsets < TOP_K
    expanded_offsets = token_idx * TOP_K + route_offsets
    permuted_indices = tl.load(
        expanded_idx_to_permuted_idx + expanded_offsets,
        mask=route_mask,
        other=-1,
    )
    scales = tl.load(
        topk_scales + expanded_offsets,
        mask=route_mask,
        other=0.0,
    ).to(tl.float32)

    tl.extra.cuda.gdc_wait()

    offsets_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
        columns = k_start + offsets_k
        column_mask = columns < HIDDEN_SIZE
        accumulator = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for route_idx in range(TOP_K):
            route_selector = tl.full((1,), route_idx, tl.int32)
            permuted_idx = tl.gather(permuted_indices, route_selector, axis=0)
            scale = tl.gather(scales, route_selector, axis=0)
            values = tl.load(
                permuted_input + permuted_idx * HIDDEN_SIZE + columns,
                mask=column_mask & (permuted_idx >= 0),
                other=0.0,
            ).to(tl.float32)
            accumulator += values * scale

        tl.store(
            output + token_idx * HIDDEN_SIZE + columns,
            accumulator,
            mask=column_mask,
        )


def moe_unpermute_bf16_pdl(
    permuted_input: torch.Tensor,
    output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    topk_scales: torch.Tensor,
    num_tokens: int,
    top_k: int,
    num_warps: int,
) -> None:
    """Combine expert-permuted BF16 rows in deterministic top-k order."""
    hidden_size = int(permuted_input.size(1))
    block_k = min(_MAX_BLOCK_K, triton.next_power_of_2(hidden_size))
    _moe_unpermute_bf16_pdl_kernel[(num_tokens,)](
        permuted_input,
        output,
        expanded_idx_to_permuted_idx,
        topk_scales,
        HIDDEN_SIZE=hidden_size,
        TOP_K=top_k,
        ROUTE_BLOCK=triton.next_power_of_2(top_k),
        BLOCK_K=block_k,
        num_warps=num_warps,
        launch_pdl=True,
    )
