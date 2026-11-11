# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
# Ported from ResInfer 51b906836a69fa43915040314df57bc9f23030ad.
# Original source: csrc/resattn/plan.py
"""Static launch selection for ResAttn decode."""

import os


def _reduction_policy(value=None):
    value = (
        os.environ.get("RESIDUE_ATTN_REDUCTION", "legacy") if value is None else value
    )
    if value not in ("legacy", "auto", "deterministic"):
        raise ValueError(f"unknown ResAttn reduction policy: {value}")
    return value


def _check_output_dimension(head_dim, output_nvfp4):
    if head_dim not in (128, 256):
        raise NotImplementedError("ResAttn supports head_dim=128 or 256")
    if head_dim == 256 and output_nvfp4:
        raise NotImplementedError("D256 supports BF16 output only")


def select_decode_plan(
    batch,
    num_kv_heads,
    max_kv_len,
    sm_count,
    *,
    output_nvfp4=False,
    grouped_head_tile=16,
    residue_fold=True,
    reduction_policy=None,
    head_dim=128,
):
    """Select the static split/reducer, then apply the caller's reduction policy."""
    _check_output_dimension(head_dim, output_nvfp4)
    policy = _reduction_policy(reduction_policy)
    if not residue_fold or grouped_head_tile not in (8, 16):
        raise NotImplementedError("production ResAttn requires residue Q8/Q16")
    splits, reducer = _select_decode_plan(
        batch, num_kv_heads, max_kv_len, sm_count, output_nvfp4
    )
    if (policy == "deterministic" or head_dim == 256) and reducer not in (
        "none",
        "kernel",
    ):
        reducer = "kernel"
    return splits, reducer


def static_compile_plans(
    *, output_nvfp4=False, grouped_head_tile=16, residue_fold=True, head_dim=128
):
    """Enumerate decode kernel compilation variants."""
    if not residue_fold or grouped_head_tile not in (8, 16):
        raise NotImplementedError("production ResAttn requires residue Q8/Q16")
    _check_output_dimension(head_dim, output_nvfp4)
    plans = [(1, "none"), (2, "kernel")]
    if head_dim == 256:
        return tuple(plans)
    if output_nvfp4:
        plans.extend(((4, "gmem_fused"), (16, "gmem_fused")))
    else:
        plans.append((4, "cga_sharded"))
    return tuple(plans)


def validate_decode_plan(plan, *, output_nvfp4=False, head_dim=128):
    """Check launch constraints before compilation or workspace allocation."""
    _check_output_dimension(head_dim, output_nvfp4)
    splits, reducer = plan
    if head_dim == 256 and reducer not in ("none", "kernel"):
        raise NotImplementedError("D256 supports none/kernel reduction only")
    if type(splits) is not int or splits < 1:
        raise ValueError(
            f"decode plan splits must be a positive integer, got {splits!r}"
        )
    if reducer not in ("none", "kernel", "cga_sharded", "gmem_fused"):
        raise ValueError(f"unsupported decode reducer: {reducer!r}")
    if (
        output_nvfp4
        and reducer == "cga_sharded"
        or not output_nvfp4
        and reducer == "gmem_fused"
    ):
        raise NotImplementedError(
            f"{reducer} does not support output_nvfp4={output_nvfp4}"
        )
    if reducer == "none" and splits != 1:
        raise ValueError("none reduction requires splits=1")
    if reducer == "cga_sharded" and splits not in (1, 2, 4, 8, 16):
        raise ValueError("cga_sharded requires splits in {1,2,4,8,16}")
    if reducer == "gmem_fused" and splits not in (4, 8, 16, 32, 64):
        raise ValueError("gmem_fused requires splits in {4,8,16,32,64}")


def _power_of_two(value):
    return 1 << (value.bit_length() - 1)


def _select_decode_plan(batch, heads, length, sms, packed):
    """Select split count and reduction mode for a decode workload."""
    grid = max(1, batch * heads)
    tiles = max(1, (length + 255) // 256)
    budget = _power_of_two(max(1, sms // grid))
    splits = min(budget, _power_of_two((tiles + 1) // 2), 64)
    if splits == 1:
        return 1, "none"
    if packed:
        splits = min(budget, max(4, _power_of_two(max(1, tiles // 4))), 16)
        return splits, "gmem_fused" if splits >= 4 else "kernel"
    if sms != 148:
        return splits, "kernel"
    cga = min(16, _power_of_two(max(2, sms // (2 * grid))), _power_of_two(tiles))
    if cga >= 2 and (length <= 2048 * cga or splits == 2):
        return cga, "cga_sharded"
    return splits, "kernel"
