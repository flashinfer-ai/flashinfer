# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Side stores shared by the existing MegaMoE forward epilogues."""

import cutlass
import cutlass.cute as cute

from flashinfer.moe_ep.batch_invariant.nvfp4_device import ScalarF32


@cute.jit
def store_preact_element(
    preact: cute.Tensor,
    expert_ids: cute.Tensor,
    gate,
    up,
    row,
    feature,
    intermediate: cutlass.Constexpr[int],
    expert,
):
    """Store one logical gate/up pair without changing its source registers."""
    if feature < intermediate:
        preact[row, feature] = gate.to(cutlass.BFloat16)
        preact[row, intermediate + feature] = up.to(cutlass.BFloat16)
        # Feature zero has exactly one writer for each valid receiver-pool row.
        if feature == 0:
            expert_ids[row] = expert


@cute.jit
def store_preact_row(
    preact: cute.Tensor,
    expert_ids: cute.Tensor,
    gate: cute.Tensor,
    up: cute.Tensor,
    row,
    feature_start,
    intermediate: cutlass.Constexpr[int],
    expert,
):
    """Save a contiguous fragment whose first feature is aligned to 16.

    Pack copies of the FP32 values into BF16 vectors so each thread writes
    complete 32-byte sectors, rather than striding across rows with scalar STG.
    The caller predicates valid pool rows; feature padding is handled here.
    """
    fragment = cute.size(gate)
    if feature_start < intermediate:
        if cutlass.const_expr(intermediate % 16 == 0 and fragment % 16 == 0):
            if feature_start + fragment <= intermediate:
                atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                    num_bits_per_copy=256,
                )
                for channel in cutlass.range_constexpr(2):
                    if cutlass.const_expr(channel == 0):
                        source = gate
                    else:
                        source = up
                    for offset in cutlass.range_constexpr(0, fragment, 16):
                        values = cute.make_tensor(
                            source.iterator + offset, cute.make_layout((16,))
                        )
                        packed = cute.make_rmem_tensor((16,), cutlass.BFloat16)
                        packed.store(values.load().to(cutlass.BFloat16))
                        pointer = (
                            preact.iterator
                            + row * (2 * intermediate)
                            + channel * intermediate
                            + feature_start
                            + offset
                        )
                        aligned = cute.make_ptr(
                            cutlass.BFloat16,
                            pointer.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        destination = cute.make_tensor(
                            aligned, cute.make_layout((16,))
                        )
                        cute.copy(atom, packed, destination)
                if feature_start == 0:
                    expert_ids[row] = expert
            else:
                for i in cutlass.range_constexpr(fragment):
                    store_preact_element(
                        preact, expert_ids, gate[i], up[i], row,
                        feature_start + i, intermediate, expert,
                    )
        else:
            for i in cutlass.range_constexpr(fragment):
                store_preact_element(
                    preact, expert_ids, gate[i], up[i], row,
                    feature_start + i, intermediate, expert,
                )


@cute.jit
def store_preact_fragment16(
    preact: cute.Tensor,
    expert_ids: cute.Tensor,
    gate: cute.Tensor,
    up: cute.Tensor,
    pool_base,
    token_start,
    valid_tokens,
    feature_start,
    lane,
    intermediate: cutlass.Constexpr[int],
    expert,
    alpha=None,
    token_scale=None,
):
    """Save the raw Ld16x64b register layout to canonical gate-then-up rows."""
    parity = (lane // 2) % 2
    feature = feature_start + (lane % 2) * 8 + lane // 4
    for i in cutlass.range_constexpr(16):
        token = token_start + 2 * i + parity
        if token < valid_tokens:
            row = pool_base + token
            g, u = gate[i], up[i]
            if cutlass.const_expr(token_scale is not None):
                scale = token_scale[row]
                if cutlass.const_expr(alpha is not None):
                    scale = ScalarF32.mul(scale, alpha)
                g = ScalarF32.mul(g, scale)
                u = ScalarF32.mul(u, scale)
            elif cutlass.const_expr(alpha is not None):
                g = ScalarF32.mul(g, alpha)
                u = ScalarF32.mul(u, alpha)
            store_preact_element(
                preact, expert_ids, g, u, row, feature, intermediate, expert,
            )
