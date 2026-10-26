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

"""Dynamic per-token E4M3 epilogue shared by the LL, BT, and HT protocols.

The recipe is the one of the TRT-LLM MNNVL ``kDynamicFP8`` RMSNorm epilogue,
and the results match it up to FP32 rounding:

* the FP32 norm (before any BF16 rounding) is the quantization input;
* ``scale = max(amax(|norm|) / 448, 2**-9 / 448)`` per token, so all-zero and
  tiny rows keep a positive scale;
* ``q = E4M3(clamp(norm * (1 / scale), -448, 448))`` with round-to-nearest;
* ``scale`` is written as FP32 to ``scale_out[token]``; ``q * scale``
  dequantizes.

The FP32 operations differ in order: the norm here is
``(x * inv_rms) * gamma``, TRT-LLM's ``(gamma * x) * inv_rms``, and TRT-LLM
divides by ``scale`` where this multiplies by a reciprocal computed once per
token (see :func:`store_dynamic_fp8_output`). A NaN stays NaN here
(``cvt.satfinite`` keeps it), whereas TRT-LLM's ``fminf``/``fmaxf`` clamp turns
it into ``-448``.

Each protocol owns the reduction of the per-thread maxima across the threads,
CTAs, or cluster that hold one token row; this module supplies the per-vector
pieces and the CTA-wide reduction used by single-CTA-per-token tails.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .cute_dsl_primitives import (
    bf16x8_to_packed_u32x4,
    map_shared_to_peer,
    store_global_u32x2,
    store_global_u32x4,
    store_shared_cluster_f32,
)
from .runtime import make_fake_dynamic_compact_tensor, to_cute_dynamic

FP8_E4M3_MAX = 448.0
# Smallest positive E4M3FN subnormal (2**-9) over the largest finite value.
DYNAMIC_FP8_MIN_SCALE = (1.0 / 512.0) / FP8_E4M3_MAX


def fake_output_scale(output_dtype: torch.dtype):
    """Compile-time stand-in for the FP8 scale operand, or ``None`` for BF16.

    The operand has a dynamic length: one element for a static per-tensor
    scale, one element per token for a dynamic ``scale_out``.
    """
    if output_dtype != torch.float8_e4m3fn:
        return None
    return make_fake_dynamic_compact_tensor(Float32, alignment=4, divisibility=1)


def output_scale_arg(output_scale: torch.Tensor | None):
    """Runtime FP8 scale operand matching :func:`fake_output_scale`."""
    if output_scale is None:
        return None
    return to_cute_dynamic(output_scale.reshape(-1), 4, divisibility=1)


@dsl_user_op
def cvt_f32x2_to_e4m3x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack two FP32 values as E4M3 (``lo`` in bits 0-7, ``hi`` in 8-15)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 fp8_pair;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, $2, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def abs_max_f32x8(values, running: Float32) -> Float32:
    """Fold ``max(|values[i]|)`` of an FP32x8 fragment into ``running``."""
    result = running
    for index in cutlass.range_constexpr(8):
        value = values[index]
        result = cute.arch.fmax(result, cute.arch.fmax(value, -value))
    return result


@cute.jit
def token_scale_from_amax(amax: Float32) -> Float32:
    return cute.arch.fmax(amax / Float32(FP8_E4M3_MAX), Float32(DYNAMIC_FP8_MIN_SCALE))


@cute.jit
def store_token_scale(scale_out: cute.Tensor, token: Int64, scale: Float32) -> None:
    cute.arch.store((scale_out.iterator + token).llvm_ptr, scale)


@cute.jit
def store_dynamic_fp8_output(
    values,
    output: cute.Tensor,
    element_offset: Int64,
    inverse_scale: Float32,
    norm_output_bf16: cute.Tensor | None,
) -> None:
    """Quantize and store eight FP32 norm values, plus the optional BF16 norm.

    ``q = E4M3(norm * (1 / scale))`` with an IEEE reciprocal computed once per
    token. TRT-LLM divides each element instead; the two agree except where
    the quotient lands within one FP32 ulp of an E4M3 rounding midpoint. A
    per-element division costs more than the whole epilogue otherwise does.
    """
    scaled = values * inverse_scale
    packed_fp8 = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
    for word in cutlass.range_constexpr(2):
        # satfinite saturates finite and infinite values to +-448 like
        # TRT-LLM's explicit clamp, but keeps NaN.
        low = cvt_f32x2_to_e4m3x2(scaled[word * 4], scaled[word * 4 + 1])
        high = cvt_f32x2_to_e4m3x2(scaled[word * 4 + 2], scaled[word * 4 + 3])
        packed_fp8[word] = low | (high << Uint32(16))
    store_global_u32x2(
        Int64((output.iterator + element_offset).toint()), packed_fp8.load()
    )
    if cutlass.const_expr(norm_output_bf16 is not None):
        store_global_u32x4(
            Int64((norm_output_bf16.iterator + element_offset).toint()),
            bf16x8_to_packed_u32x4(values.to(BFloat16)),
        )


@cute.jit
def cta_max(
    value: Float32,
    warp_maxima: cute.Tensor,
    warps: cutlass.Constexpr[int],
) -> Float32:
    """Maximum of ``value`` over every thread of the CTA.

    ``warp_maxima`` is a shared array of ``warps`` FP32 slots that no other
    reduction uses; every thread receives the result.
    """
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    value = cute.arch.warp_reduction_max(value)
    if lane == 0:
        cute.arch.store((warp_maxima + warp).llvm_ptr, value)
    cute.arch.barrier()
    result = Float32(0.0)
    for slot in cutlass.range_constexpr(warps):
        result = cute.arch.fmax(
            result, cute.arch.load((warp_maxima + Int32(slot)).llvm_ptr, Float32)
        )
    return result


@cute.jit
def store_dynamic_fp8_cta_row(
    prenorm_fragments: cute.Tensor,
    gamma_fragments: cute.Tensor,
    inverse_rms: Float32,
    norm_output: cute.Tensor,
    norm_output_bf16: cute.Tensor | None,
    scale_out: cute.Tensor,
    warp_maxima: cute.Tensor,
    tidx: Int32,
    token: Int32,
    hidden: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    weight_bias: cutlass.Constexpr[float],
) -> None:
    """Dynamic E4M3 epilogue for a token row owned by one CTA.

    Thread ``tidx`` holds the BF16x8 fragments ``tidx + trip * threads`` of
    the pre-norm row and of gamma in ``(trips, 8)`` register tensors. The FP32
    norm is recomputed in the second pass instead of being kept live across
    the reduction; both passes evaluate the same expression.
    """
    fragments = hidden // 8
    trips = (fragments + threads - 1) // threads
    thread_max = Float32(0.0)
    for trip in cutlass.range_constexpr(trips):
        fragment = tidx + trip * threads
        if fragment < fragments:
            gamma_value = gamma_fragments[trip, None].load().to(Float32)
            if cutlass.const_expr(weight_bias != 0.0):
                gamma_value = gamma_value + Float32(weight_bias)
            thread_max = abs_max_f32x8(
                prenorm_fragments[trip, None].load().to(Float32)
                * inverse_rms
                * gamma_value,
                thread_max,
            )
    token_scale = token_scale_from_amax(cta_max(thread_max, warp_maxima, threads // 32))
    if tidx == 0:
        store_token_scale(scale_out, Int64(token), token_scale)
    inverse_scale = Float32(1.0) / token_scale
    for trip in cutlass.range_constexpr(trips):
        fragment = tidx + trip * threads
        if fragment < fragments:
            gamma_value = gamma_fragments[trip, None].load().to(Float32)
            if cutlass.const_expr(weight_bias != 0.0):
                gamma_value = gamma_value + Float32(weight_bias)
            store_dynamic_fp8_output(
                prenorm_fragments[trip, None].load().to(Float32)
                * inverse_rms
                * gamma_value,
                norm_output,
                Int64(token) * hidden + Int64(fragment) * 8,
                inverse_scale,
                norm_output_bf16,
            )


@cute.jit
def store_dynamic_fp8_cluster_row(
    prenorm_fragments: cute.Tensor,
    gamma_fragments: cute.Tensor,
    inverse_rms: Float32,
    norm_output: cute.Tensor,
    norm_output_bf16: cute.Tensor | None,
    scale_out: cute.Tensor,
    warp_maxima: cute.Tensor,
    cluster_maxima: cute.Tensor,
    tidx: Int32,
    token: Int32,
    base_fragment: Int32,
    rank_lane: Int32,
    cluster_rank: Int32,
    hidden: cutlass.Constexpr[int],
    fragment_stride: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    cluster_size: cutlass.Constexpr[int],
    weight_bias: cutlass.Constexpr[float],
) -> None:
    """Dynamic E4M3 epilogue for a token row spread over a CTA cluster.

    Rank-lane leaders hold the BF16x8 fragments
    ``base_fragment + trip * fragment_stride``; other lanes contribute a zero
    maximum. The CTA maxima are exchanged through distributed shared memory,
    so the caller must already have synchronized the cluster once (every peer
    CTA is running). ``warp_maxima`` and ``cluster_maxima`` are shared arrays
    that no other reduction uses.
    """
    fragments = hidden // 8
    trips = (fragments + fragment_stride - 1) // fragment_stride
    thread_max = Float32(0.0)
    for trip in cutlass.range_constexpr(trips):
        fragment = base_fragment + trip * fragment_stride
        if fragment < fragments and rank_lane == 0:
            gamma_value = gamma_fragments[trip, None].load().to(Float32)
            if cutlass.const_expr(weight_bias != 0.0):
                gamma_value = gamma_value + Float32(weight_bias)
            thread_max = abs_max_f32x8(
                prenorm_fragments[trip, None].load().to(Float32)
                * inverse_rms
                * gamma_value,
                thread_max,
            )
    block_max = cta_max(thread_max, warp_maxima, threads // 32)
    if tidx < cluster_size:
        store_shared_cluster_f32(
            map_shared_to_peer(cluster_maxima + cluster_rank, Int32(tidx)), block_max
        )
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    token_max = Float32(0.0)
    for peer in cutlass.range_constexpr(cluster_size):
        token_max = cute.arch.fmax(
            token_max, cute.arch.load((cluster_maxima + peer).llvm_ptr, Float32)
        )
    token_scale = token_scale_from_amax(token_max)
    if cluster_rank == 0 and tidx == 0:
        store_token_scale(scale_out, Int64(token), token_scale)
    inverse_scale = Float32(1.0) / token_scale
    for trip in cutlass.range_constexpr(trips):
        fragment = base_fragment + trip * fragment_stride
        if fragment < fragments and rank_lane == 0:
            gamma_value = gamma_fragments[trip, None].load().to(Float32)
            if cutlass.const_expr(weight_bias != 0.0):
                gamma_value = gamma_value + Float32(weight_bias)
            store_dynamic_fp8_output(
                prenorm_fragments[trip, None].load().to(Float32)
                * inverse_rms
                * gamma_value,
                norm_output,
                Int64(token) * hidden + Int64(fragment) * 8,
                inverse_scale,
                norm_output_bf16,
            )
