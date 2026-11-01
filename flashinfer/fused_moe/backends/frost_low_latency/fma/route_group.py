# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse weight-only FC1 weights across dynamically matched expert routes."""

import cutlass
from cutlass import cute
from .bf16.fma_fc1 import _quad, _single_route_body
from .activations import apply_activation
from .fp4_weight import scaled_quad, weight_scale


@cute.jit
def load_weight(
    wp,
    sf,
    expert,
    row,
    col,
    rows: cutlass.Constexpr,
    width: cutlass.Constexpr,
    native: cutlass.Constexpr,
    scale_block: cutlass.Constexpr,
    fp8_weights: cutlass.Constexpr,
):
    word = wp[(expert * rows + row) * (width // 4) + col]
    if cutlass.const_expr(fp8_weights):
        return _quad(word, cutlass.Float8E4M3FN)
    scale = weight_scale(sf[expert, row, col // (scale_block // 4)], scale_block)
    return scaled_quad(word, scale, native, scale_block)


@cute.kernel
def _kernel(
    x,
    w,
    sf,
    dq,
    ids,
    mid,
    columns: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    native: cutlass.Constexpr,
    scale_block: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
):
    bid, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    h = cute.size(x, mode=[1])
    i = cute.size(mid, mode=[1])
    k = cute.size(ids, mode=[1])
    routes = cute.size(ids)
    tiles = i // (8 * columns)
    leader = bid // tiles
    row = (bid % tiles) * 8 * columns + warp * columns
    expert = ids[leader // k, leader % k].to(cutlass.Int64)
    branches = 2 if gated else 1
    fp8_weights = w.element_type == cutlass.Float8E4M3FN
    if expert >= 0 and expert < cute.size(w, mode=[0]):
        # Each group is owned by its first occurrence in flattened route order.
        # All lanes in a warp see the same leader; ballot operations are uniform.
        rank = cutlass.Int32(0)
        total = cutlass.Int32(0)
        masks = cute.make_rmem_tensor((routes + 31) // 32, cutlass.Uint32)
        for chunk in cutlass.range_constexpr((routes + 31) // 32):
            r = chunk * 32 + lane
            hit = cutlass.Boolean(False)
            if r < routes:
                hit = ids[r // k, r % k].to(cutlass.Int64) == expert
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
        if rank % group_size == 0:
            # Avoid the larger register footprint when no weight can be shared.
            if total - rank == 1:
                _single_route_body(
                    x,
                    w,
                    ids,
                    mid,
                    columns,
                    activation,
                    gated,
                    parameters,
                    dq,
                    None,
                    sf,
                    native,
                    scale_block,
                )
            else:
                members = cute.make_rmem_tensor(group_size, cutlass.Int32)
                members.fill(-1)
                offset = cutlass.Int32(0)
                lane_mask = (cutlass.Uint32(1) << lane) - cutlass.Uint32(1)
                for chunk in cutlass.range_constexpr((routes + 31) // 32):
                    r = chunk * 32 + lane
                    mask = masks[chunk]
                    limit = leader - chunk * 32
                    if limit >= 32:
                        mask = cutlass.Uint32(0)
                    elif limit > 0:
                        mask = mask & (
                            ~((cutlass.Uint32(1) << limit) - cutlass.Uint32(1))
                        )
                    hit = ((mask >> lane) & cutlass.Uint32(1)) != 0
                    position = offset + cute.arch.popc(mask & lane_mask)
                    for p in cutlass.range_constexpr(group_size):
                        owner = cute.arch.vote_ballot_sync(hit & (position == p)).to(
                            cutlass.Uint32
                        )
                        if owner != 0:
                            src = cute.arch.popc(
                                (owner & (cutlass.Uint32(0) - owner))
                                - cutlass.Uint32(1)
                            )
                            members[p] = cute.arch.shuffle_sync(r, src)
                    offset = offset + cute.arch.popc(mask)
                up = cute.make_rmem_tensor((group_size, 4, columns), cutlass.Float32)
                up.fill(0.0)
                gate = cute.make_rmem_tensor((group_size, 4, columns), cutlass.Float32)
                gate.fill(0.0)
                xv = cute.make_rmem_tensor((group_size, 4), cutlass.Float32)
                xp = cute.recast_ptr(x.iterator, dtype=cutlass.Uint64)
                wp = cute.recast_ptr(
                    w.iterator, dtype=cutlass.Uint32 if fp8_weights else cutlass.Uint16
                )
                for group in range(h // 128):
                    col = group * 32 + lane
                    xv.fill(0.0)
                    for p in cutlass.range_constexpr(group_size):
                        if members[p] >= 0:
                            values = _quad(
                                xp[(members[p] // k) * (h // 4) + col], x.element_type
                            )
                            for c in cutlass.range_constexpr(4):
                                xv[p, c] = values[c]
                    for o in cutlass.range_constexpr(columns):
                        uv = load_weight(
                            wp,
                            sf,
                            expert,
                            row + o,
                            col,
                            branches * i,
                            h,
                            native,
                            scale_block,
                            fp8_weights,
                        )
                        if cutlass.const_expr(gated):
                            gv = load_weight(
                                wp,
                                sf,
                                expert,
                                row + o + i,
                                col,
                                branches * i,
                                h,
                                native,
                                scale_block,
                                fp8_weights,
                            )
                        for p in cutlass.range_constexpr(group_size):
                            if members[p] >= 0:
                                for c in cutlass.range_constexpr(4):
                                    up[p, c, o] = up[p, c, o] + xv[p, c] * uv[c].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(gated):
                                        gate[p, c, o] = gate[p, c, o] + xv[p, c] * gv[
                                            c
                                        ].to(cutlass.Float32)
                for p in cutlass.range_constexpr(group_size):
                    for o in cutlass.range_constexpr(columns):
                        u = cute.arch.warp_reduction_sum(
                            up[p, 0, o] + up[p, 1, o] + up[p, 2, o] + up[p, 3, o]
                        )
                        g = cutlass.Float32(0.0)
                        if cutlass.const_expr(gated):
                            g = cute.arch.warp_reduction_sum(
                                gate[p, 0, o]
                                + gate[p, 1, o]
                                + gate[p, 2, o]
                                + gate[p, 3, o]
                            )
                        if lane == 0 and members[p] >= 0:
                            u = (
                                (u * (dq[expert] if fp8_weights else dq[expert, 0]))
                                .to(mid.element_type)
                                .to(cutlass.Float32)
                            )
                            if cutlass.const_expr(gated):
                                g = (
                                    (g * (dq[expert] if fp8_weights else dq[expert, 1]))
                                    .to(mid.element_type)
                                    .to(cutlass.Float32)
                                )
                            mid[members[p], row + o] = apply_activation(
                                u, g, activation, parameters=parameters
                            ).to(mid.element_type)
    else:
        if lane == 0:
            for o in cutlass.range_constexpr(columns):
                mid[leader, row + o] = mid.element_type(0.0)


# Custom activation limits may stringify with a '+' that NVVM rejects in names.
_kernel.set_name_prefix(
    "flashinfer_fma_route_group_fc1",
    keep_mangled_name=False,
    remove_cutlass_symbol=True,
)
