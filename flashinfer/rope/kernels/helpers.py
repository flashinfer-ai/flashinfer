"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

RoPE Computation Helpers for CuTe-DSL Kernels
=============================================

This module provides helper functions for RoPE computation:
- apply_rope_interleaved_fp16/bf16: Apply RoPE to interleaved (GPT-J style) pairs
- apply_rope_non_interleaved_fp16/bf16: Apply RoPE to non-interleaved (NeoX style) pairs
- compute_llama31_freq: Compute Llama 3.1 frequency scaling
"""

import cutlass
from cutlass import Float32, Uint32

from .ptx_ops import (
    half2_to_float2,
    float2_to_half2,
    bfloat2_to_float2,
    float2_to_bfloat2,
    clamp_f32,
)


def apply_rope_interleaved_fp16(
    v: Uint32, sin_val: Float32, cos_val: Float32
) -> Uint32:
    """Apply interleaved RoPE to a half2 pair (fp16).

    For interleaved mode, pairs are adjacent: (x0, x1) where:
    - y0 = x0 * cos - x1 * sin
    - y1 = x0 * sin + x1 * cos
    """
    x0, x1 = half2_to_float2(v)
    y0 = x0 * cos_val - x1 * sin_val
    y1 = x0 * sin_val + x1 * cos_val
    return float2_to_half2(y0, y1)


def apply_rope_interleaved_bf16(
    v: Uint32, sin_val: Float32, cos_val: Float32
) -> Uint32:
    """Apply interleaved RoPE to a bfloat2 pair (bf16).

    For interleaved mode, pairs are adjacent: (x0, x1) where:
    - y0 = x0 * cos - x1 * sin
    - y1 = x0 * sin + x1 * cos
    """
    x0, x1 = bfloat2_to_float2(v)
    y0 = x0 * cos_val - x1 * sin_val
    y1 = x0 * sin_val + x1 * cos_val
    return float2_to_bfloat2(y0, y1)


def apply_rope_non_interleaved_fp16(
    v: Uint32,
    p: Uint32,
    sin0: Float32,
    cos0: Float32,
    sin1: Float32,
    cos1: Float32,
    rope_sign: Float32,
) -> Uint32:
    """Apply non-interleaved RoPE to a half2 pair (fp16).

    For non-interleaved mode, pairs are separated by half_rotary_dim:
    - If in first half: y = x * cos - pair * sin (rope_sign = -1)
    - If in second half: y = x * cos + pair * sin (rope_sign = +1)

    Parameters
    ----------
    v : Uint32
        Main vector (2 fp16 elements packed)
    p : Uint32
        Pair vector (2 fp16 elements at offset +/- half_rotary_dim)
    sin0, cos0 : Float32
        Sin/cos for first element
    sin1, cos1 : Float32
        Sin/cos for second element
    rope_sign : Float32
        -1.0 if in first half, +1.0 if in second half
    """
    x0, x1 = half2_to_float2(v)
    px0, px1 = half2_to_float2(p)
    y0 = x0 * cos0 + rope_sign * px0 * sin0
    y1 = x1 * cos1 + rope_sign * px1 * sin1
    return float2_to_half2(y0, y1)


def apply_rope_non_interleaved_bf16(
    v: Uint32,
    p: Uint32,
    sin0: Float32,
    cos0: Float32,
    sin1: Float32,
    cos1: Float32,
    rope_sign: Float32,
) -> Uint32:
    """Apply non-interleaved RoPE to a bfloat2 pair (bf16).

    For non-interleaved mode, pairs are separated by half_rotary_dim:
    - If in first half: y = x * cos - pair * sin (rope_sign = -1)
    - If in second half: y = x * cos + pair * sin (rope_sign = +1)

    Parameters
    ----------
    v : Uint32
        Main vector (2 bf16 elements packed)
    p : Uint32
        Pair vector (2 bf16 elements at offset +/- half_rotary_dim)
    sin0, cos0 : Float32
        Sin/cos for first element
    sin1, cos1 : Float32
        Sin/cos for second element
    rope_sign : Float32
        -1.0 if in first half, +1.0 if in second half
    """
    x0, x1 = bfloat2_to_float2(v)
    px0, px1 = bfloat2_to_float2(p)
    y0 = x0 * cos0 + rope_sign * px0 * sin0
    y1 = x1 * cos1 + rope_sign * px1 * sin1
    return float2_to_bfloat2(y0, y1)


def compute_llama31_freq(
    base_freq: Float32,
    smooth_a: Float32,
    smooth_b: Float32,
    rope_rcp_scale: Float32,
) -> Float32:
    """
    Apply Llama 3.1 frequency scaling.

    Formula:
        smooth = clamp(base_freq * smooth_a + smooth_b, 0, 1)
        freq = (1 - smooth) * (base_freq * rope_rcp_scale) + smooth * base_freq

    This smoothly interpolates between scaled and unscaled frequencies based on
    the frequency value. Low frequencies are scaled more, high frequencies less.

    Parameters
    ----------
    base_freq : Float32
        Base frequency value (1/theta^(2i/d))
    smooth_a : Float32
        Smoothing coefficient a = old_context_len / (2*pi*(high_freq_factor - low_freq_factor))
    smooth_b : Float32
        Smoothing coefficient b = -low_freq_factor / (high_freq_factor - low_freq_factor)
    rope_rcp_scale : Float32
        Reciprocal of rope_scale (1/rope_scale)

    Returns
    -------
    Float32
        Scaled frequency value
    """
    smooth = clamp_f32(base_freq * smooth_a + smooth_b, Float32(0.0), Float32(1.0))
    # Lerp: (1-t)*a + t*b = a + t*(b-a)
    scaled_freq = base_freq * rope_rcp_scale
    return scaled_freq + smooth * (base_freq - scaled_freq)


def apply_rope_interleaved_dispatch(
    v: Uint32,
    sin_val: Float32,
    cos_val: Float32,
    is_fp16: bool,
) -> Uint32:
    """Dispatch to fp16 or bf16 interleaved RoPE based on compile-time flag.

    This helper reduces code duplication by encapsulating the cutlass.const_expr()
    dispatch pattern.

    Parameters
    ----------
    v : Uint32
        Input vector (2 elements packed)
    sin_val : Float32
        Sin value
    cos_val : Float32
        Cos value
    is_fp16 : bool
        Compile-time flag: True for fp16, False for bf16

    Returns
    -------
    Uint32
        Rotated vector
    """
    if cutlass.const_expr(is_fp16):
        return apply_rope_interleaved_fp16(v, sin_val, cos_val)
    else:
        return apply_rope_interleaved_bf16(v, sin_val, cos_val)


def apply_rope_non_interleaved_dispatch(
    v: Uint32,
    p: Uint32,
    sin0: Float32,
    cos0: Float32,
    sin1: Float32,
    cos1: Float32,
    rope_sign: Float32,
    is_fp16: bool,
) -> Uint32:
    """Dispatch to fp16 or bf16 non-interleaved RoPE based on compile-time flag.

    This helper reduces code duplication by encapsulating the cutlass.const_expr()
    dispatch pattern.

    Parameters
    ----------
    v : Uint32
        Main vector (2 elements packed)
    p : Uint32
        Pair vector (2 elements packed)
    sin0, cos0 : Float32
        Sin/cos for first element
    sin1, cos1 : Float32
        Sin/cos for second element
    rope_sign : Float32
        -1.0 if in first half, +1.0 if in second half
    is_fp16 : bool
        Compile-time flag: True for fp16, False for bf16

    Returns
    -------
    Uint32
        Rotated vector
    """
    if cutlass.const_expr(is_fp16):
        return apply_rope_non_interleaved_fp16(v, p, sin0, cos0, sin1, cos1, rope_sign)
    else:
        return apply_rope_non_interleaved_bf16(v, p, sin0, cos0, sin1, cos1, rope_sign)


__all__ = [
    "apply_rope_interleaved_fp16",
    "apply_rope_interleaved_bf16",
    "apply_rope_non_interleaved_fp16",
    "apply_rope_non_interleaved_bf16",
    "apply_rope_interleaved_dispatch",
    "apply_rope_non_interleaved_dispatch",
    "compute_llama31_freq",
]
