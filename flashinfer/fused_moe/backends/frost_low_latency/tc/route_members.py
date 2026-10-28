# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live route membership, reusing Frost FC1 group ownership."""

import cutlass
from cutlass import cute


@cute.jit
def route_members(ids, leader, group_size: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    routes = cute.size(ids)
    expert = ids[leader].to(cutlass.Int64)
    rank = cutlass.Int32(0)
    total = cutlass.Int32(0)
    masks = cute.make_rmem_tensor((routes + 31) // 32, cutlass.Uint32)
    for chunk in cutlass.range_constexpr((routes + 31) // 32):
        r = chunk * 32 + lane
        hit = cutlass.Boolean(False)
        if r < routes:
            hit = ids[r].to(cutlass.Int64) == expert
        mask = cute.arch.vote_ballot_sync(hit).to(cutlass.Uint32)
        masks[chunk] = mask
        total = total + cute.arch.popc(mask)
        before = cutlass.Uint32(0)
        limit = leader - chunk * 32
        if limit >= 32:
            before = mask
        elif limit > 0:
            before = mask & ((cutlass.Uint32(1) << limit) - cutlass.Uint32(1))
        rank = rank + cute.arch.popc(before)
    members = cute.make_rmem_tensor(group_size, cutlass.Int32)
    members.fill(-1)
    if rank % group_size == 0:
        offset = cutlass.Int32(0)
        lane_mask = (cutlass.Uint32(1) << lane) - cutlass.Uint32(1)
        for chunk in cutlass.range_constexpr((routes + 31) // 32):
            r = chunk * 32 + lane
            mask = masks[chunk]
            limit = leader - chunk * 32
            if limit >= 32:
                mask = cutlass.Uint32(0)
            elif limit > 0:
                mask = mask & (~((cutlass.Uint32(1) << limit) - cutlass.Uint32(1)))
            hit = ((mask >> lane) & cutlass.Uint32(1)) != 0
            position = offset + cute.arch.popc(mask & lane_mask)
            for p in cutlass.range_constexpr(group_size):
                owner = cute.arch.vote_ballot_sync(hit & (position == p)).to(
                    cutlass.Uint32
                )
                if owner != 0:
                    src = cute.arch.popc(
                        (owner & (cutlass.Uint32(0) - owner)) - cutlass.Uint32(1)
                    )
                    members[p] = cute.arch.shuffle_sync(r, src)
            offset = offset + cute.arch.popc(mask)
    return members
