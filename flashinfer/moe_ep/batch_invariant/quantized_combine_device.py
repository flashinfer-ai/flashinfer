# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Ordered, masked reduction of quantized MegaMoE combine slots."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import vector
from cutlass.cutlass_dsl import T

from .nvfp4_device import PackedF32
from .wire_device import ScalarF32
from .wire_spec import (
    combine_amax,
    combine_decode_scale,
    combine_quantize_amax_precast,
    combine_quantize_block_precast,
)


@cute.jit
def _store_packed(data, blocks):
    values = []
    for block_idx in cutlass.range_constexpr(len(blocks)):
        words = blocks[block_idx].bitcast(cutlass.Int32)
        for i in cutlass.range_constexpr(cute.size(words.shape)):
            values.append(words[i].ir_value())
    packed = vector.from_elements(T.vector(len(values), T.i32()), values)
    cute.recast_tensor(data, cutlass.Int32).store(
        cute.TensorSSA(packed, (len(values),), cutlass.Int32)
    )


@cute.jit
def quantize_combine_wire(
    prequant_reg, *, combine_dtype, across_lanes=False, lane_idx=None
):
    """Encode the same logical BF16 blocks for STG and UBLK carriers."""
    vec = 16 if combine_dtype == "nvfp4" else 32
    data_dtype = (
        cutlass.Float4E2M1FN if combine_dtype == "nvfp4" else cutlass.Float8E4M3FN
    )
    sf_dtype = cutlass.BFloat16 if combine_dtype == "nvfp4" else cutlass.Float8E8M0FNU
    size = cute.size(prequant_reg)
    data = cute.make_rmem_tensor((size,), data_dtype)
    sf = cute.make_rmem_tensor((size // vec,), sf_dtype)
    blocks = []
    if cutlass.const_expr(across_lanes):
        lane_in_block = lane_idx % cutlass.Int32(vec)
        first_lane = lane_idx - lane_in_block
        scaled_regs = cute.make_rmem_tensor((size,), cutlass.Float32)
        for token in cutlass.range_constexpr(size):
            value = ScalarF32.widen(prequant_reg[token])
            values = tuple(
                cute.arch.shuffle_sync(value, first_lane + j) for j in range(vec)
            )
            maximum = combine_amax(ScalarF32, values)
            scaled, stored_scale = combine_quantize_amax_precast(
                ScalarF32, (value,), maximum, combine_dtype=combine_dtype
            )
            scaled_regs[token] = scaled[0]
            if token % vec == lane_in_block:
                sf[token // vec] = stored_scale
        blocks.append(scaled_regs.load().to(data_dtype))
    else:
        for block in cutlass.range_constexpr(size // vec):
            values = tuple(
                ScalarF32.widen(prequant_reg[block * vec + j]) for j in range(vec)
            )
            scaled, stored_scale = combine_quantize_block_precast(
                ScalarF32, values, combine_dtype=combine_dtype
            )
            scaled_regs = cute.make_rmem_tensor((vec,), cutlass.Float32)
            for j in cutlass.range_constexpr(vec):
                scaled_regs[j] = scaled[j]
            blocks.append(scaled_regs.load().to(data_dtype))
            sf[block] = stored_scale
    _store_packed(data, blocks)
    return data, sf


@cute.jit
def ordered_quantized_slot_fold(
    codes,
    scales,
    topk_scores,
    topk_indices,
    token_idx,
    *,
    num_experts,
    hidden_per_thread,
    num_topk,
    act_dtype,
    decode_fp4,
):
    """Fold every logical slot, using zero operands for an invalid route.

    ``codes`` and ``scales`` are the caller's logical (slot, hidden) views;
    scales broadcast within their wire-format block. FP4 decoding stays packed
    and uses the provider's native conversion helper. All arithmetic after
    decoding is shared across static and persistent scheduling paths.
    """
    acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
    acc.fill(0)
    load_bits = min(128, hidden_per_thread * act_dtype.width)
    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), act_dtype, num_bits_per_copy=load_bits
    )
    for slot in cutlass.range_constexpr(num_topk):
        value = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
        value.fill(0)
        scale = cutlass.Float32(0)
        score = cutlass.Float32(0)
        valid_route = True
        if cutlass.const_expr(topk_indices is not None):
            expert = topk_indices[token_idx, cutlass.Int32(slot)]
            valid_route = (expert >= 0) & (expert < num_experts)
        if valid_route:
            term = cute.make_rmem_tensor((hidden_per_thread,), act_dtype)
            source = codes[slot, None]
            pointer = source.iterator
            source = cute.make_tensor(
                cute.make_ptr(
                    pointer.dtype,
                    pointer.toint(),
                    pointer.memspace,
                    assumed_align=hidden_per_thread * act_dtype.width // 8,
                ),
                source.layout,
            )
            cute.copy(load_atom, source, term)
            if cutlass.const_expr(act_dtype is cutlass.Float4E2M1FN):
                decode_fp4(term, value)
                scale = combine_decode_scale(
                    ScalarF32, scales[slot, 0], combine_dtype="nvfp4"
                )
            else:
                value.store(term.load().to(cutlass.Float32))
                scale = combine_decode_scale(
                    ScalarF32, scales[slot, 0], combine_dtype="mxfp8"
                )
            if cutlass.const_expr(topk_scores is not None):
                score = cutlass.Float32(topk_scores[token_idx, cutlass.Int32(slot)])
        for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
            decoded = PackedF32.mul((value[i], value[i + 1]), (scale, scale))
            previous = (acc[i], acc[i + 1])
            if cutlass.const_expr(topk_scores is None):
                acc[i], acc[i + 1] = PackedF32.add(previous, decoded)
            else:
                acc[i], acc[i + 1] = PackedF32.fma(decoded, (score, score), previous)
    return acc
