# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Softmax / P-compute helpers for FMHA decode TS resources.

Used by ``TmemSResource`` (softmax reduction, atomic-max scratch),
``SmemPResource`` (S→P conversion and quantization), ``TmemCorrResource``
(attention-sink normalization), and the softmax-stats resources.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from cutlass.experimental import primitives as prims

from .....device.lse_merge import merge_log2_lse
from ..config import FmhaDecodeConfig
from .common import Constexpr, fadd2, ffma2
from .q_rows import q_row_token_and_local_head
from .....device.packing import pack_float4_to_fp8_e4m3
from cutlass.experimental.task_scheduling.resources import StageInfo


@cute.jit
def attention_sink_for_local_head(
    cfg: FmhaDecodeConfig,
    attention_sinks_ptr: cute.Pointer | None,
    scale_softmax_log2: Float32,
    max_val: Float32,
    logical_h_k_idx: Int32,
    h_r: Int32,
    num_heads_kv: Int32,
    local_head_idx: Int32,
) -> Float32:
    """Return the attention-sink denominator contribution for one local head."""
    # Attention sinks add a synthetic key/value entry to the normalization
    # denominator. Return zero when the feature is disabled so call sites can
    # use the same reduction flow.
    if cutlass.const_expr(not cfg.use_attention_sinks):
        return Float32(0.0)
    head_idx = cute.math.min(
        logical_h_k_idx * h_r + local_head_idx,
        h_r * num_heads_kv - Int32(1),
    )
    sink_ptr = cutlass.inttoptr(
        attention_sinks_ptr.toint() + cutlass.Int64(head_idx * Int32(4)),
        mem_space=1,
        dtype=Float32,
    )
    sink_val = sink_ptr.load(count=1, alignment=4)[0]
    sink_exp = cute.math.exp2(
        sink_val * Float32(1.4426950408889634) - max_val * scale_softmax_log2,
        fastmath=True,
    )
    if cutlass.const_expr(cfg.use_fp8_pv):
        sink_exp = sink_exp * Float32(cfg.fp8_p_quant_scale)
    return sink_exp


@cute.jit
def attention_sink_for_scale_idx(
    cfg: FmhaDecodeConfig,
    attention_sinks_ptr: cute.Pointer | None,
    scale_softmax_log2: Float32,
    max_val: Float32,
    logical_h_k_idx: Int32,
    h_r: Int32,
    num_heads_kv: Int32,
    logical_q_group_idx: Int32,
    col_group_idx: Int32,
    scale_idx: Constexpr[int],
) -> Float32:
    """Map a softmax scale group to its attention-sink contribution."""
    tile_row_idx = (
        Int32((scale_idx // 2) * 8) + col_group_idx * Int32(2) + Int32(scale_idx % 2)
    )
    _, local_head_idx = q_row_token_and_local_head(
        cfg, h_r, logical_q_group_idx, tile_row_idx
    )
    head_stride = h_r
    if cutlass.const_expr(cfg.heads_q_per_kv > 0):
        # Multi-token profiles flatten token/head rows differently, but
        # attention sinks remain one value per physical Q head. Resolve the
        # CTA row through the shared Q geometry and stride by the true ratio.
        head_stride = Int32(cfg.heads_q_per_kv)
    return attention_sink_for_local_head(
        cfg,
        attention_sinks_ptr,
        scale_softmax_log2,
        max_val,
        logical_h_k_idx,
        head_stride,
        num_heads_kv,
        local_head_idx,
    )


@cute.jit
def combine_int_frac_ex2(
    x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None
) -> Float32:
    """Scale ``frac_ex2`` by ``2**floor(x)`` through the FP32 exponent bits.

    ``x_rounded`` still carries the magic rounding constant, so its low
    mantissa bits hold ``floor(x)``; shifting them into the exponent field and
    adding the bits of the polynomial result multiplies by the integer power.
    """
    return cute.arch.inline_ptx(
        "{\n"
        "  .reg .b32 xi;\n"
        "  .reg .b32 fi;\n"
        "  .reg .b32 xe;\n"
        "  .reg .b32 oi;\n"
        "  mov.b32 xi, {$r0};\n"
        "  mov.b32 fi, {$r1};\n"
        "  shl.b32 xe, xi, 23;\n"
        "  add.s32 oi, xe, fi;\n"
        "  mov.b32 {$w0}, oi;\n"
        "}",
        write_only_types=[Float32],
        read_only_args=[x_rounded, frac_ex2],
        loc=loc,
        ip=ip,
    )


@cute.jit
def compute_fp8_p_regs_and_local_sums(
    scale_softmax_log2: Float32,
    addend_0: Float32,
    addend_1: Float32,
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
) -> tuple[Int32, Int32, Float32, Float32]:
    """Exponentiate two scale groups of four scores into packed FP8 P and sums.

    Every probability is ``exp2(c * s + addend)`` with the group's exponent
    addend (``SmemPResource.exponent_addend``), so no mask logic is needed:
    a masked score at ``-FLT_MAX`` exponentiates to zero.
    """
    addend_pair = (addend_0, addend_1)
    scaled_pair_01 = ffma2(
        (s0, s1),
        (scale_softmax_log2, scale_softmax_log2),
        addend_pair,
    )
    scaled_pair_23 = ffma2(
        (s2, s3),
        (scale_softmax_log2, scale_softmax_log2),
        addend_pair,
    )
    prob0 = cute.math.exp2(scaled_pair_01[0], fastmath=True)
    prob1 = cute.math.exp2(scaled_pair_01[1], fastmath=True)
    scaled_pair_45 = ffma2(
        (s4, s5),
        (scale_softmax_log2, scale_softmax_log2),
        addend_pair,
    )
    prob2 = cute.math.exp2(scaled_pair_23[0], fastmath=True)
    prob3 = cute.math.exp2(scaled_pair_23[1], fastmath=True)
    # Accumulate the local softmax sums while packing P for the BMM2 operand.
    local_sum_pair_01 = fadd2((prob0, prob1), (prob2, prob3))
    packed_p_0 = pack_float4_to_fp8_e4m3(prob0, prob1, prob2, prob3)
    scaled_pair_67 = ffma2(
        (s6, s7),
        (scale_softmax_log2, scale_softmax_log2),
        addend_pair,
    )
    prob4 = cute.math.exp2(scaled_pair_45[0], fastmath=True)
    prob5 = cute.math.exp2(scaled_pair_45[1], fastmath=True)
    prob6 = cute.math.exp2(scaled_pair_67[0], fastmath=True)
    prob7 = cute.math.exp2(scaled_pair_67[1], fastmath=True)
    local_sum_pair_45 = fadd2((prob4, prob5), (prob6, prob7))
    packed_p_1 = pack_float4_to_fp8_e4m3(prob4, prob5, prob6, prob7)
    local_sum_pair = fadd2(local_sum_pair_01, local_sum_pair_45)

    return packed_p_0, packed_p_1, local_sum_pair[0], local_sum_pair[1]


@cute.jit
def compute_p_values_and_local_sums(
    scale_softmax_log2: Float32,
    addend_0: Float32,
    addend_1: Float32,
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
) -> tuple[
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
]:
    """Exponentiate two scale groups of four scores into FP32 P and paired sums.

    Straight-line twin of ``_compute_fp8_p_regs_and_local_sums`` for 16-bit
    P: the same ``exp2(c * s + addend)`` per score, returned unpacked so the
    caller can choose the P storage format.
    """
    addend_pair = (addend_0, addend_1)
    scaled_pair_01 = ffma2(
        (s0, s1),
        (scale_softmax_log2, scale_softmax_log2),
        addend_pair,
    )
    scaled_pair_23 = ffma2(
        (s2, s3),
        (scale_softmax_log2, scale_softmax_log2),
        addend_pair,
    )
    prob0 = cute.math.exp2(scaled_pair_01[0], fastmath=True)
    prob1 = cute.math.exp2(scaled_pair_01[1], fastmath=True)
    scaled_pair_45 = ffma2(
        (s4, s5),
        (scale_softmax_log2, scale_softmax_log2),
        addend_pair,
    )
    prob2 = cute.math.exp2(scaled_pair_23[0], fastmath=True)
    prob3 = cute.math.exp2(scaled_pair_23[1], fastmath=True)
    scaled_pair_67 = ffma2(
        (s6, s7),
        (scale_softmax_log2, scale_softmax_log2),
        addend_pair,
    )
    prob4 = cute.math.exp2(scaled_pair_45[0], fastmath=True)
    prob5 = cute.math.exp2(scaled_pair_45[1], fastmath=True)
    prob6 = cute.math.exp2(scaled_pair_67[0], fastmath=True)
    prob7 = cute.math.exp2(scaled_pair_67[1], fastmath=True)
    local_sum_pair_02 = fadd2((prob0, prob1), (prob2, prob3))
    local_sum_pair_46 = fadd2((prob4, prob5), (prob6, prob7))
    local_sum_pair = fadd2(local_sum_pair_02, local_sum_pair_46)
    return (
        prob0,
        prob1,
        prob2,
        prob3,
        prob4,
        prob5,
        prob6,
        prob7,
        local_sum_pair[0],
        local_sum_pair[1],
    )


@cute.jit
def ex2_emulation_packed_f32x2(x: Float32, y: Float32) -> tuple[Float32, Float32]:
    """Evaluate ``2**x`` and ``2**y`` on the FMA pipe instead of MUFU.

    Inputs are non-positive scaled scores minus the row maximum. The integer
    part is split off with a magic-constant rounding add, the fraction in
    [0, 1) goes through a degree-3 minimax polynomial, and the two parts are
    recombined through the exponent bits. The relative error stays below the
    BF16 rounding of the P operand, matching the emulation used by the dense
    Blackwell FMHA kernels.
    """
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm"
    )
    xy_rounded_back = cute.arch.sub_packed_f32x2(
        xy_rounded, (fp32_round_int, fp32_round_int)
    )
    xy_frac = cute.arch.sub_packed_f32x2(xy_clamped, xy_rounded_back)
    coeff = (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    )
    out = (coeff[3], coeff[3])
    for degree in cutlass.range_constexpr(2, -1, -1):
        out = cute.arch.fma_packed_f32x2(out, xy_frac, (coeff[degree], coeff[degree]))
    return (
        combine_int_frac_ex2(xy_rounded[0], out[0]),
        combine_int_frac_ex2(xy_rounded[1], out[1]),
    )


@cute.jit
def final_output_lse(cfg, sink_ptr, lse_ptr, mass_lse, output_row, head, writer):
    """Merge a sink once in log space and publish unamplified final LSE.

    Input mass includes the internal FP8 P amplification. Partial statistics
    remain in that domain; only the externally visible final store removes it.
    """
    weight = Float32(1.0)
    if cutlass.const_expr(cfg.use_attention_sinks or cfg.lse_mode != "none"):
        value = mass_lse
        if cutlass.const_expr(cfg.use_fp8_pv):
            value -= Float32(cfg.fp8_p_quant_log2_scale)
        if cutlass.const_expr(cfg.use_attention_sinks):
            sink = (sink_ptr + cutlass.Int64(head)).load() * Float32(1.4426950408889634)
            value, weight, _ = merge_log2_lse(value, sink)
        if cutlass.const_expr(cfg.lse_mode != "none"):
            if writer:
                (lse_ptr + cutlass.Int64(output_row)).store(value)
    return weight


@cute.jit
def masked_weighted_sum(terms: tuple) -> Float32:
    """Return the sum of ``value * weight`` over four ``(uses, value, weight)`` terms.

    A term whose ``uses`` is false contributes zero, whatever its value. The
    fixed term order makes lanes that share a row agree bitwise.
    """
    assert len(terms) == 4
    (uses0, value0, weight0), (uses1, value1, weight1) = terms[0], terms[1]
    (uses2, value2, weight2), (uses3, value3, weight3) = terms[2], terms[3]
    masked0 = Float32(0.0)
    masked1 = Float32(0.0)
    masked2 = Float32(0.0)
    masked3 = Float32(0.0)
    if uses0:
        masked0 = value0
    if uses1:
        masked1 = value1
    if uses2:
        masked2 = value2
    if uses3:
        masked3 = value3
    lanes = ffma2((masked0, masked2), (weight0, weight2), (Float32(0.0), Float32(0.0)))
    lanes = ffma2((masked1, masked3), (weight1, weight3), lanes)
    return lanes[0] + lanes[1]


def softmax_scale_pair_width(num_scale_groups: int, scale_base: int) -> int:
    """Return the number of live lanes in one packed two-group operation."""
    return min(2, num_scale_groups - scale_base)


def softmax_tile_idx(
    cfg: FmhaDecodeConfig, stage_info: StageInfo, inst_id: int
) -> Int32:
    """Tile index consumed by the softmax-side MMA loop (inst_id ∈ {0, 1})."""
    return stage_info.loop_offset * Int32(cfg.num_insts_kv) + Int32(inst_id)


@cute.jit
def wspro_reduce_max4(
    val0: Float32,
    val1: Float32,
    val2: Float32,
    val3: Float32,
    local_row_idx: Int32,
) -> Float32:
    """Reduce four independent maxima across four strided warp rows."""
    # The four column groups are interleaved every four lanes. The conditional
    # swaps transpose independent scale groups into row ownership before each
    # full-warp butterfly; every lane executes all three shuffle operations.
    left01 = val0
    right01 = val1
    left23 = val2
    right23 = val3
    if (local_row_idx & Int32(1)) == Int32(0):
        tmp = left01
        left01 = right01
        right01 = tmp
        tmp = left23
        left23 = right23
        right23 = tmp
    left01 = Float32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=left01,
            offset=4,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    )
    left23 = Float32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=left23,
            offset=4,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    )
    reduced01 = cute.math.max(left01, right01, ftz=True)
    reduced23 = cute.math.max(left23, right23, ftz=True)

    if local_row_idx < Int32(2):
        tmp = reduced01
        reduced01 = reduced23
        reduced23 = tmp
    reduced01 = Float32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=reduced01,
            offset=8,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    )
    return cute.math.max(reduced01, reduced23, ftz=True)
