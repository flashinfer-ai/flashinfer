# SPDX-License-Identifier: Apache-2.0
"""Fused Rubin staging for MXFP4 routed and MXFP8 shared experts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_byte(amax, reciprocal_limit: tl.constexpr):
    scale = tl.maximum(amax * reciprocal_limit, 2.0**-126)
    bits = scale.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    return tl.minimum(exponent, 254)


@triton.jit(do_not_specialize=["live_rows", "capacity", "scale_rows"])
def _stage(
    x,
    ids,
    scores,
    routed_x,
    routed_sf,
    shared_x,
    shared_sf,
    out_ids,
    out_scores,
    live_rows,
    capacity,
    scale_rows,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_E4M3: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    k_tile = tl.program_id(1)
    cols = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    live = rows < live_rows
    data_mask = live[:, None] & (cols[None, :] < HIDDEN)

    GROUPS: tl.constexpr = BLOCK_K // 32
    SCALE_COLS: tl.constexpr = HIDDEN // 32
    groups = k_tile * GROUPS + tl.arange(0, GROUPS)
    shared_exp = tl.full((BLOCK_M, GROUPS), 0, tl.uint32)
    shared_data = tl.full((BLOCK_M, BLOCK_K), 0, tl.uint8)
    if row_start < live_rows:
        values = tl.load(
            x + rows[:, None] * HIDDEN + cols[None, :], data_mask, other=0
        ).to(tl.float32)
        values = tl.reshape(values, (BLOCK_M, GROUPS, 32))
        amax = tl.max(tl.abs(values), axis=2)
        routed_exp = _scale_byte(amax, 1.0 / 6.0)
        routed_inv = ((254 - routed_exp) << 23).to(tl.float32, bitcast=True)
        even, odd = tl.split(
            tl.reshape(values * routed_inv[:, :, None], (BLOCK_M, GROUPS, 16, 2))
        )
        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
            constraints="=r,f,f",
            args=[even, odd],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        pairs = k_tile * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        tl.store(
            routed_x + rows[:, None] * (HIDDEN // 2) + pairs[None, :],
            tl.reshape(packed, (BLOCK_M, BLOCK_K // 2)).to(tl.uint8),
            live[:, None] & (pairs[None, :] < HIDDEN // 2),
        )
        tl.store(
            routed_sf + rows[:, None] * SCALE_COLS + groups[None, :],
            routed_exp.to(tl.uint8),
            live[:, None] & (groups[None, :] < SCALE_COLS),
        )

        # The shared Torch quantizer maps nonfinite maxima to the normal floor.
        shared_amax = tl.where(amax < float("inf"), amax, 0.0)
        if SHARED_E4M3:
            shared_exp = _scale_byte(shared_amax, 1.0 / 448.0)
        else:
            shared_exp = _scale_byte(shared_amax, 1.0 / 57344.0)
        shared_inv = ((254 - shared_exp) << 23).to(tl.float32, bitcast=True)
        scaled = tl.reshape(values * shared_inv[:, :, None], (BLOCK_M, BLOCK_K))
        if SHARED_E4M3:
            shared_data = scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        else:
            shared_data = scaled.to(tl.float8e5).to(tl.uint8, bitcast=True)

    tl.store(
        shared_x + rows[:, None] * HIDDEN + cols[None, :],
        tl.where(live[:, None], shared_data, 0),
        (rows[:, None] < capacity) & (cols[None, :] < HIDDEN),
    )

    # Write the 128-row x 4-column scale atom in its final 32x4x4 order.
    atom = (rows[:, None] // 128) * (SCALE_COLS // 4) + groups[None, :] // 4
    in_atom = (rows[:, None] % 32) * 16 + ((rows[:, None] % 128) // 32) * 4
    tl.store(
        shared_sf + atom * 512 + in_atom + groups[None, :] % 4,
        tl.where(live[:, None], shared_exp, 0).to(tl.uint8),
        (rows[:, None] < scale_rows) & (groups[None, :] < SCALE_COLS),
    )

    if k_tile == 0:
        slots = tl.arange(0, triton.next_power_of_2(TOP_K))
        offsets = rows[:, None] * TOP_K + slots[None, :]
        valid = live[:, None] & (slots[None, :] < TOP_K)
        expert = tl.load(ids + offsets, valid, other=-1)
        weight = tl.load(scores + offsets, valid, other=0)
        tl.store(
            out_ids + offsets,
            expert.to(tl.int32),
            (rows[:, None] < capacity) & (slots[None, :] < TOP_K),
        )
        tl.store(out_scores + offsets, weight, valid)


def stage_routed_and_shared(tensors, workspace):
    """Quantize one BF16 input into both expert layouts and clear stale routes."""
    x = tensors.hidden_states
    rows, hidden = x.shape
    capacity = workspace.x.shape[0]
    scale_rows = triton.cdiv(capacity, 128) * 128
    # Wider tiles amortize scheduling and vectorize stores at high capacity.
    # A long zero tail benefits from more rows per CTA.
    if capacity <= 128:
        block_m, block_k = 4, 256
    else:
        block_m, block_k = (16 if rows * 8 < capacity else 8), 512
    _stage[(triton.cdiv(scale_rows, block_m), triton.cdiv(hidden, block_k))](
        x,
        tensors.topk_ids,
        tensors.topk_weights,
        workspace.x.view(torch.uint8),
        workspace.x_sf.view(torch.uint8),
        workspace.shared_x.view(torch.uint8),
        workspace.shared_x_sf.view(torch.uint8),
        workspace.topk_idx,
        workspace.topk_weights,
        rows,
        capacity,
        scale_rows,
        hidden,
        tensors.topk_ids.shape[1],
        workspace.config.shared_expert.kind == "mxfp8_e4m3",
        block_m,
        block_k,
        num_warps=4,
    )
