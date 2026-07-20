# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/integration/triton_route.py @ a1e5af4e (2026-03-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _route_topk_kernel(
    logits_ptr,
    topk_logits_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    logits_row_stride,
    topk_row_stride,
    num_experts,
    BLOCK_E: tl.constexpr,
    TOP_K: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_offsets_i32 = expert_offsets.to(tl.int32)
    topk_offsets = tl.arange(0, TOPK_BLOCK)
    topk_mask = topk_offsets < TOP_K
    neg_inf = float("-inf")

    row_ptr = logits_ptr + pid * logits_row_stride
    logits = tl.load(
        row_ptr + expert_offsets, mask=expert_offsets < num_experts, other=neg_inf
    )
    remaining = logits.to(tl.float32)

    topk_logits = tl.full((TOPK_BLOCK,), neg_inf, tl.float32)
    topk_ids = tl.zeros((TOPK_BLOCK,), dtype=tl.int32)

    for slot in range(TOP_K):
        best_logits = tl.max(remaining, axis=0)
        best_ids = tl.max(
            tl.where(remaining == best_logits, expert_offsets_i32, -1),
            axis=0,
        )
        topk_logits = tl.where(topk_offsets == slot, best_logits, topk_logits)
        topk_ids = tl.where(topk_offsets == slot, best_ids, topk_ids)
        remaining = tl.where(expert_offsets == best_ids, neg_inf, remaining)

    if RENORMALIZE:
        masked_logits = tl.where(topk_mask, topk_logits, neg_inf)
        shifted = masked_logits - tl.max(masked_logits, axis=0)
        exp_logits = tl.where(topk_mask, tl.exp(shifted), 0.0)
        topk_weights = exp_logits / tl.sum(exp_logits, axis=0)
    else:
        topk_weights = topk_logits

    out_base = pid * topk_row_stride + topk_offsets
    tl.store(topk_logits_ptr + out_base, topk_logits, mask=topk_mask)
    tl.store(topk_ids_ptr + out_base, topk_ids, mask=topk_mask)
    tl.store(topk_weights_ptr + out_base, topk_weights, mask=topk_mask)


def route_topk(
    router_logits: torch.Tensor,
    topk_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    renormalize: bool,
) -> None:
    if router_logits.ndim != 2:
        raise ValueError(
            "expected router_logits with rank 2, got shape "
            f"{tuple(router_logits.shape)}"
        )
    if topk_logits.ndim != 2 or topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("top-k outputs must all have rank 2")
    if topk_logits.shape != topk_ids.shape or topk_logits.shape != topk_weights.shape:
        raise ValueError(
            "top-k outputs must share a shape, got "
            f"{tuple(topk_logits.shape)}, {tuple(topk_ids.shape)}, {tuple(topk_weights.shape)}"
        )
    if router_logits.shape[0] != topk_logits.shape[0]:
        raise ValueError(
            "router_logits batch mismatch: expected "
            f"{router_logits.shape[0]}, got {topk_logits.shape[0]}"
        )
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"expected topk_ids dtype int32, got {topk_ids.dtype}")
    if topk_logits.dtype != torch.float32:
        raise ValueError(f"expected topk_logits dtype float32, got {topk_logits.dtype}")
    if topk_weights.dtype != torch.float32:
        raise ValueError(
            f"expected topk_weights dtype float32, got {topk_weights.dtype}"
        )
    if not router_logits.is_cuda:
        raise ValueError("route_topk requires CUDA tensors")
    if router_logits.shape[1] > 1024:
        raise ValueError(
            f"route_topk currently supports up to 1024 experts, got {router_logits.shape[1]}"
        )
    if router_logits.stride(-1) != 1:
        raise ValueError("router_logits must be contiguous in the expert dimension")
    if (
        topk_logits.stride(-1) != 1
        or topk_ids.stride(-1) != 1
        or topk_weights.stride(-1) != 1
    ):
        raise ValueError("top-k outputs must be contiguous in the top-k dimension")

    num_tokens, num_experts = router_logits.shape
    top_k = topk_logits.shape[1]
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")

    block_e = triton.next_power_of_2(num_experts)
    topk_block = triton.next_power_of_2(top_k)
    num_warps = 4 if block_e <= 256 else 8

    _route_topk_kernel[(num_tokens,)](
        router_logits,
        topk_logits,
        topk_ids,
        topk_weights,
        router_logits.stride(0),
        topk_logits.stride(0),
        num_experts,
        BLOCK_E=block_e,
        TOP_K=top_k,
        TOPK_BLOCK=topk_block,
        RENORMALIZE=renormalize,
        num_warps=num_warps,
    )
