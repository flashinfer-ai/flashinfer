# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Decode completed protocol records independently of the CUDA implementation."""

import torch


def moe_previous_sender(records, q_len, receiver, generation):
    sender = (receiver - 1) % 8
    slot = sender if sender < receiver else sender - 1
    words = records[: q_len * 0x38000].view(torch.int32).reshape(q_len, 7, 2, 128, 8, 4)
    payload = words[:, slot, generation & 1]
    assert bool((payload[..., 3] == generation).all()), "stale/incomplete MoE record"
    partial = payload[..., :3].contiguous().view(torch.bfloat16).reshape(q_len, 6144)
    return sender, partial.cpu()


def attention_rank0_partials(records, q_len, generation):
    if q_len == 4:
        words = (
            records[:917504]
            .view(torch.int32)
            .reshape(2, 7, 128, 4, 8, 4)[generation & 1]
        )
        assert bool((words[..., 3] == generation).all()), (
            "stale/incomplete q4 attention record"
        )
        partials = (
            words[..., :3].contiguous().view(torch.bfloat16).reshape(7, 128, 4, 8, 6)
        )
        return partials.permute(0, 2, 1, 3, 4).reshape(7, 4, 6144).contiguous().cpu()
    words = (
        records[:229376].view(torch.int32).reshape(7, 2, 128, 8, 4)[:, generation & 1]
    )
    assert bool((words[..., 3] == generation).all()), (
        "stale/incomplete q1 attention record"
    )
    return words[..., :3].contiguous().view(torch.bfloat16).reshape(7, 1, 6144).cpu()
