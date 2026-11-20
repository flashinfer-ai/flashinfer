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

"""Reusable low-level operations for MLA decode TS kernels."""

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.experimental import primitives as cprims

from .constants import WARP_REDUCTION_BFLY_DISTANCES


"""Public primitive inline-PTX entry point used by MLA helper ops."""

inline_ptx = cute.arch.inline_ptx
"""CuTe inline PTX entry point used by MLA helper ops."""


@cutlass.dsl_user_op
def vector_from_scalars(values, dtype, *, loc=None, ip=None):
    """Pack scalar register values into a DSL vector."""
    return cutlass.Vector.from_elements(
        tuple(dtype(value) for value in values),
        dtype,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def fmax_f32(a, b, *, loc=None, ip=None):
    """Return the maximum of two values as Float32."""
    return Float32(
        cute.math.max(Float32(a), Float32(b), ftz=True, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def warp_reduce_max_f32(val, *, loc=None, ip=None):
    """Reduce a Float32 value to the warp maximum with butterfly shuffles."""
    val = Float32(val)
    for dist in WARP_REDUCTION_BFLY_DISTANCES:
        val = fmax_f32(
            val,
            Float32(
                cprims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=val,
                    offset=dist,
                    mask_and_clamp=0x1F,
                    kind=cprims.Shfl.BFLY,
                    loc=loc,
                    ip=ip,
                )
            ),
            loc=loc,
            ip=ip,
        )
    return val


@cutlass.dsl_user_op
def warp_reduce_sum_f32(val, *, loc=None, ip=None):
    """Reduce a Float32 value to the warp sum with butterfly shuffles."""
    val = Float32(val)
    for dist in WARP_REDUCTION_BFLY_DISTANCES:
        val = Float32(
            val
            + cprims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=val,
                offset=dist,
                mask_and_clamp=0x1F,
                kind=cprims.Shfl.BFLY,
                loc=loc,
                ip=ip,
            )
        )
    return val


@cute.jit
def pack_float4_to_fp8_e4m3(v0: Float32, v1: Float32, v2: Float32, v3: Float32):
    """Pack four Float32 values into one E4M3 x4 register."""
    # Spell the canonical pair conversions through CuTe's public inline-PTX
    # operation so the generated f8x2 instruction receives only its supported
    # operands.
    return inline_ptx(
        "{\n"
        "  .reg .b16 lo;\n"
        "  .reg .b16 hi;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};\n"
        "  cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};\n"
        "  mov.b32 {$w0}, {lo, hi};\n"
        "}",
        write_only_types=[Int32],
        read_only_args=[v0, v1, v2, v3],
    )


@cute.jit
def fp8_log2_quant_scale():
    """Return log2(448) for the E4M3 P scaling convention."""
    return Float32(8.8073549)


@cute.jit
def fp8_quant_scale_rcp():
    """Return the reciprocal of the shared E4M3 probability scale."""
    return Float32(1.0 / 448.0)


# A positive ``softmax_rescale_threshold_log2`` T keeps a row's max anchor while
# a K tile raises it by at most T binades, so P reaches 2**T. E4M3 P then uses
# the scale 448 * 2**-T, as Prims-TS FMHA decode does (``fp8_p_quant_scale``),
# so the scaled P stays within 448. This equals the trtllm-gen FMHA
# ``getSoftmaxPScale`` at T = 0 (448) and at T = 8 (1.75). T is a compile-time
# config value, so these are plain Python helpers rather than ``cute.jit``
# functions.
def _fp8_rescale_threshold_p_scale(cfg) -> float:
    return 448.0 * 2.0**-cfg.softmax_rescale_threshold_log2


def fp8_log2_p_scale(cfg):
    """Return log2 of the E4M3 P scale selected by the config."""
    if cfg.softmax_rescale_threshold_log2 > 0.0:
        return Float32(math.log2(_fp8_rescale_threshold_p_scale(cfg)))
    return fp8_log2_quant_scale()


def fp8_p_scale_rcp(cfg):
    """Return the reciprocal of ``fp8_p_scale``."""
    if cfg.softmax_rescale_threshold_log2 > 0.0:
        return Float32(1.0 / _fp8_rescale_threshold_p_scale(cfg))
    return fp8_quant_scale_rcp()
