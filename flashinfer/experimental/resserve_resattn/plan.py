# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Static launch selection for ResAttn decode."""

import os


def _reduction_policy(value=None):
    value = (
        os.environ.get("RESIDUE_ATTN_REDUCTION", "legacy") if value is None else value
    )
    if value not in ("legacy", "auto", "deterministic"):
        raise ValueError(f"unknown ResAttn reduction policy: {value}")
    return value


def select_decode_plan(
    batch,
    num_kv_heads,
    max_kv_len,
    sm_count,
    *,
    grouped_head_tile=16,
    reduction_policy=None,
    head_dim=128,
):
    """Select the static split/reducer, then apply the caller's reduction policy."""
    if head_dim not in (128, 256):
        raise NotImplementedError("ResAttn supports head_dim=128 or 256")
    policy = _reduction_policy(reduction_policy)
    if grouped_head_tile not in (8, 16):
        raise NotImplementedError("production ResAttn requires residue Q8/Q16")
    splits, reducer = _select_decode_plan(batch, num_kv_heads, max_kv_len, sm_count)
    if policy == "deterministic" and reducer not in ("none", "kernel"):
        reducer = "kernel"
    return splits, reducer


def _power_of_two(value):
    return 1 << (value.bit_length() - 1)


def _select_decode_plan(batch, heads, length, sms):
    """Select split count and reduction mode for a decode workload."""
    grid = max(1, batch * heads)
    tiles = max(1, (length + 255) // 256)
    budget = _power_of_two(max(1, sms // grid))
    splits = min(budget, _power_of_two((tiles + 1) // 2), 64)
    if splits == 1:
        return 1, "none"
    if sms != 148:
        return splits, "kernel"
    cga = min(16, _power_of_two(max(2, sms // (2 * grid))), _power_of_two(tiles))
    if cga >= 2 and (length <= 2048 * cga or splits == 2):
        return cga, "cga_sharded"
    return splits, "kernel"
