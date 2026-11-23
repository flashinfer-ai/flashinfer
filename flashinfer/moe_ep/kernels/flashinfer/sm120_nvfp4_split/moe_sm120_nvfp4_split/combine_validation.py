# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Post-launch RLC transport audit. Never call inside a captured/timed region."""

import torch

from .combine_reference import check_staging_delivery, reconstruct_owner_from_staging
from .config import RANK_COMBINE_MAX_GROUPS


def rank_local_staging_report(tester) -> dict:
    """Actual staging oracle, not an independent FC1/FC2 arithmetic reference.

    Current route must have been gathered after the last replay. Return data
    mismatches without raising: the caller agrees across ranks before failing.
    """
    t = tester
    kernel = t._kernel
    if kernel is None or t.local_workspace is None:
        raise RuntimeError("staging validation requires a completed launch")
    if t._tp_reduce_enabled():
        return {"status": "NOT_CHECKED", "reason": "post-TP output is not a local K3 output"}

    def local(name, dtype, *, reshape=True):
        spec = kernel._local_region_by_name[name]
        offset = kernel._local_offsets[name]
        value = t.local_workspace[offset:offset + spec.nbytes].view(dtype)
        return value.reshape(spec.shape) if reshape else value

    mapping = local("rank_combine_route_map", torch.int32).long()
    staged = local("rank_combine_route_output", torch.bfloat16)[:, 0]
    # Metadata is declared as [capacity, 8] UInt8, not capacity Int64.
    metadata = local("token_src_metadata", torch.int64, reshape=False).flatten()
    tokens = t.problem.num_tokens_per_rank
    total = t.world_size * tokens
    routes = t._global_topk_idx.reshape(total, t.problem.num_topk)
    partial, errors = reconstruct_owner_from_staging(
        staged, mapping, metadata, routes, owner=t.rank,
        tokens_per_rank=tokens, experts_per_rank=t.problem.num_experts_per_rank,
    )
    expected_count = (routes // t.problem.num_experts_per_rank == t.rank).sum(1).int()
    max_groups = RANK_COMBINE_MAX_GROUPS
    m, n, _ = t.k2_mma_tiler_mnk
    groups = max_groups if (
        t.problem.hidden >= 2048
        and t.problem.hidden % (max_groups * max(128, m)) == 0
        and n >= max_groups
    ) else 1
    done = local("rank_combine_tile_done", torch.int32).flatten()
    capacity = kernel.pool_token_capacity
    errors["group_counters"] = 0
    for group in range(max_groups):
        expected = expected_count if group < groups else torch.zeros_like(expected_count)
        actual = done[capacity + group * total:capacity + (group + 1) * total]
        errors["group_counters"] += int((actual != expected).sum())
    joined = done[capacity + max_groups * total:capacity + (max_groups + 1) * total]
    # With a single group the kernel skips the cross-group join entirely.
    expected_join = (expected_count > 0).int() * (groups if groups > 1 else 0)
    errors["owner_join"] = int((joined != expected_join).sum())

    spec = kernel._shared_region_by_name["rank_combine_route_count"]
    offset = kernel._shared_offsets["rank_combine_route_count"]
    sideband = t.shared_workspace[offset:offset + spec.nbytes].view(torch.int32)
    errors["route_counts"] = int((sideband[:total] != expected_count).sum())
    if sideband.numel() > total + t.world_size:
        plan = torch.zeros(total, dtype=torch.int32, device=routes.device)
        count = torch.zeros_like(plan)
        for slot in range(t.problem.num_topk):
            belongs = routes[:, slot] // t.problem.num_experts_per_rank == t.rank
            plan.bitwise_or_(torch.where(belongs, slot << (count * 3), 0))
            count.add_(belongs.int())
        plan.bitwise_or_(count << 24)
        errors["contributor_plan"] = int(
            (sideband[total + t.world_size:2 * total + t.world_size] != plan).sum()
        )

    partials = [torch.empty_like(partial) for _ in range(t.world_size)]
    if t.world_size == 1:
        partials[0].copy_(partial)
    else:
        t._ep_all_gather(partials, partial)
    start = t.rank * tokens
    expected = torch.stack([p[start:start + tokens] for p in partials], dim=1)
    owners = t.my_topk_idx // t.problem.num_experts_per_rank
    active = torch.stack([(owners == owner).any(1) for owner in range(t.world_size)], dim=1)
    epoch = int(t.rank_combine_ready[-1, 0])
    errors.update(check_staging_delivery(
        expected, t.combine_output[:, :t.world_size], t.output_tensor,
        t.rank_combine_ready[:-1], active=active, epoch=epoch,
    ))
    return {
        "status": "PASS" if not any(errors.values()) else "FAIL",
        "source": "actual_FC2_BF16_staging", "certifies_FC1_FC2_arithmetic": False,
        "epoch": epoch, "groups": groups, "errors": errors,
    }
