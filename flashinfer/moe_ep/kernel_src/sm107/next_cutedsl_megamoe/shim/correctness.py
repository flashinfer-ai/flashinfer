# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded-memory, collective Torch oracle for offline qualification."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .kernel_helpers import compute_megamoe_reference_sm107_block_scaled


def sampled_reference(
    symm_buffer, l1, l2, num_tokens: int, *, process_group=None, sample_size=64
):
    """Reference evenly spaced rows including both ends, using actual staged bytes.

    Each rank evaluates its local experts for one source rank at a time.
    Communication and FP32 expert scratch are bounded by ``sample_size`` and
    one expert, independent of the session's token capacity and expert count.
    """
    cfg = symm_buffer.config
    count = min(sample_size, num_tokens)
    indices = torch.linspace(0, max(num_tokens - 1, 0), count, device="cuda").long()
    local = [
        torch.zeros(
            (sample_size, *t.shape[1:]), dtype=torch.uint8, device=t.device
        ).view(t.dtype)
        if t.element_size() == 1
        else torch.zeros((sample_size, *t.shape[1:]), dtype=t.dtype, device=t.device)
        for t in (
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )
    ]
    local[2].fill_(-1)
    for source, target in zip(
        (
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        ),
        local,
        strict=False,
    ):
        if source.element_size() == 1:
            target[:count].view(torch.uint8).copy_(source.view(torch.uint8)[indices])
        else:
            target[:count].copy_(source[indices])
    expected = None
    for source_rank in range(cfg.world_size):
        payload = [t.clone() for t in local]
        if cfg.world_size > 1:
            global_source = (
                dist.get_global_rank(process_group, source_rank)
                if process_group is not None
                else source_rank
            )
            for tensor in payload:
                dist.broadcast(
                    tensor.view(torch.uint8), src=global_source, group=process_group
                )
        partial = compute_megamoe_reference_sm107_block_scaled(
            payload[0],
            payload[1],
            payload[2],
            payload[3],
            l1[0].permute(0, 2, 1),
            l1[1],
            l2[0].permute(0, 2, 1),
            l2[1],
            quant_kind=cfg.quant_kind,
            local_expert_offset=cfg.rank * cfg.experts_per_rank,
            gate_up_clamp=cfg.gate_up_clamp,
            apply_topk_at_fc1=cfg.apply_topk_at_fc1,
            weight_scales_are_swizzled=True,
            return_fp32=True,
        )
        if cfg.world_size > 1:
            dist.reduce(
                partial, dst=global_source, op=dist.ReduceOp.SUM, group=process_group
            )
        if source_rank == cfg.rank:
            expected = partial[:count].to(torch.bfloat16)
    return indices, expected


def output_error(output, indices, expected) -> float:
    """Relative L2 on sampled rows; any nonfinite output is a failed result."""
    if not bool(torch.isfinite(output).all()):
        return float("inf")
    actual = output[indices].float()
    target = expected.float()
    denominator = target.norm().clamp_min(1e-12)
    return float((actual - target).norm() / denominator)
