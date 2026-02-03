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

RoPE (Rotary Positional Embeddings) using CuTe-DSL
==================================================

High-performance RoPE kernel implemented using CuTe-DSL.
This is an alternative backend to the CUDA C++ implementation.

Supports both interleaved (GPT-J style) and non-interleaved (NeoX style) modes.
"""

import functools
from typing import Callable, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .utils import get_cutlass_dtype


# =============================================================================
# Constants
# =============================================================================

COPY_BITS = 128  # 128-bit vectorized loads


# =============================================================================
# PTX Intrinsics for sin/cos and math computation
# =============================================================================


@dsl_user_op
def sin_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute approximate sin using PTX sin.approx.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "sin.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cos_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute approximate cos using PTX cos.approx.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cos.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ex2_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute 2^x using PTX ex2.approx.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "ex2.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def lg2_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute log2(x) using PTX lg2.approx.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "lg2.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def powf_approx(base: Float32, exp: Float32) -> Float32:
    """Compute base^exp using 2^(exp * log2(base))."""
    return ex2_approx(exp * lg2_approx(base))


@dsl_user_op
def sincos_approx(x: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """
    Compute approximate sin and cos simultaneously using PTX.

    This is more efficient than calling sin_approx and cos_approx separately
    as it computes both values in a single operation.

    Returns (sin_val, cos_val).
    """
    # PTX doesn't have a combined sincos.approx, so we compute separately
    # but this function serves as documentation that both are needed
    # and allows future optimization if PTX adds such instruction
    sin_val = Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "sin.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    cos_val = Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cos.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return sin_val, cos_val


@dsl_user_op
def fmaxf(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max(a, b) for float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fminf(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute min(a, b) for float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def clamp_f32(val: Float32, lo: Float32, hi: Float32) -> Float32:
    """Clamp val to [lo, hi]."""
    return fmaxf(lo, fminf(hi, val))


# =============================================================================
# PTX Intrinsics for Vectorized Memory Access (128-bit)
# =============================================================================


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64 for PTX instructions."""
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@dsl_user_op
def ld_global_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """
    Load 128 bits (4 x uint32) from global memory.

    For fp16: loads 8 elements (4 half2 pairs)
    For bf16: loads 8 elements (4 bfloat2 pairs)
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)

    return Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)


@dsl_user_op
def st_global_v4_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    Store 128 bits (4 x uint32) to global memory.

    For fp16: stores 8 elements (4 half2 pairs)
    For bf16: stores 8 elements (4 bfloat2 pairs)
    """
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# Half2 / BFloat2 Conversion Intrinsics for RoPE
# =============================================================================


@dsl_user_op
def half2_to_float2(h2: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Convert half2 (2 x fp16 packed in uint32) to 2 x float32."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(h2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 h0, h1;
            mov.b32 {h0, h1}, $2;
            cvt.f32.f16 $0, h0;
            cvt.f32.f16 $1, h1;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


@dsl_user_op
def float2_to_half2(f0: Float32, f1: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert 2 x float32 to half2 (2 x fp16 packed in uint32)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 h0, h1;
                cvt.rn.f16.f32 h0, $1;
                cvt.rn.f16.f32 h1, $2;
                mov.b32 $0, {h0, h1};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_to_float2(bf2: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Convert bfloat2 (2 x bf16 packed in uint32) to 2 x float32."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


@dsl_user_op
def float2_to_bfloat2(f0: Float32, f1: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert 2 x float32 to bfloat2 (2 x bf16 packed in uint32)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                shr.b32 lo, $1, 16;
                and.b32 hi, $2, 0xFFFF0000;
                or.b32 $0, lo, hi;
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Helper function to apply RoPE rotation
# =============================================================================


def apply_rope_element(
    x_val: Float32,
    pair_val: Float32,
    freq: Float32,
    pos: Int32,
    sign: Float32,
) -> Float32:
    """Apply RoPE rotation to a single element."""
    embed = Float32(pos) * freq
    sin_val = sin_approx(embed)
    cos_val = cos_approx(embed)
    return x_val * cos_val + sign * pair_val * sin_val


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

    This smoothly interpolates between scaled and unscaled frequencies.
    """
    smooth = clamp_f32(base_freq * smooth_a + smooth_b, Float32(0.0), Float32(1.0))
    # Lerp: (1-t)*a + t*b = a + t*(b-a)
    scaled_freq = base_freq * rope_rcp_scale
    return scaled_freq + smooth * (base_freq - scaled_freq)


# =============================================================================
# RoPE Kernel Classes
# =============================================================================


class RopeKernelNonInterleavedVec:
    """
    Vectorized RoPE kernel (non-interleaved/NeoX style).

    Performance optimizations (mirrors CUDA C++ vec_apply_llama_rope_cos_sin):
    1. Uses 128-bit vectorized loads for BOTH the main vector AND pair vector
    2. Determines pair offset ONCE per thread (not per element)
    3. All vec_size elements pair element-by-element after the two vector loads

    Key insight from CUDA C++:
    - Thread loads vec from `elem_offset`
    - Thread loads pair_vec from `elem_offset ± half_rotary` (sign based on which half)
    - Elements vec[i] and pair_vec[i] are paired for RoPE computation

    This achieves coalesced memory access even for non-interleaved mode!
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        head_dim: int,
        rotary_dim: int,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.half_rotary = rotary_dim // 2

        # Vectorized: each thread handles 8 elements (4 half2 pairs)
        # 128 bits = 4 x u32 = 8 x fp16
        self.elems_per_thread = 8
        self.bdx = head_dim // self.elems_per_thread
        self.num_threads = max(128, self.bdx)
        self.bdy = self.num_threads // self.bdx

        # Determine if fp16 or bf16
        self.is_fp16 = dtype.width == 16 and dtype == cutlass.Float16

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        q_rope: cute.Tensor,
        k_rope: cute.Tensor,
        pos_ids: cute.Tensor,
        nnz: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
        rope_rcp_scale: Float32,
        rope_rcp_theta: Float32,
        smooth_a: Float32,
        smooth_b: Float32,
        stream,
    ):
        tokens_per_block = self.bdy
        num_token_blocks = (nnz + tokens_per_block - 1) // tokens_per_block
        total_heads = num_qo_heads + num_kv_heads

        self.kernel(
            q,
            k,
            q_rope,
            k_rope,
            pos_ids,
            nnz,
            num_qo_heads,
            num_kv_heads,
            rope_rcp_scale,
            rope_rcp_theta,
            smooth_a,
            smooth_b,
        ).launch(
            grid=[num_token_blocks, total_heads, 1],
            block=[self.bdx, self.bdy, 1],
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        q_rope: cute.Tensor,
        k_rope: cute.Tensor,
        pos_ids: cute.Tensor,
        nnz: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
        rope_rcp_scale: Float32,
        rope_rcp_theta: Float32,
        smooth_a: Float32,
        smooth_b: Float32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        head_dim = self.head_dim
        rotary_dim = self.rotary_dim
        half_rotary = self.half_rotary
        elems_per_thread = self.elems_per_thread
        bdy = self.bdy
        is_fp16 = self.is_fp16

        token_idx = bidx * bdy + tidy
        elem_offset = tidx * elems_per_thread

        # Determine if this block handles Q or K
        is_q_head = bidy < num_qo_heads
        head_idx = bidy
        k_head_idx = bidy - num_qo_heads

        # Key insight: determine pair offset and sign ONCE per thread (like CUDA C++)
        # If elem_offset < half_rotary: pair is at elem_offset + half_rotary, sign = -1
        # Else: pair is at elem_offset - half_rotary, sign = +1
        in_first_half = elem_offset < half_rotary

        # Pre-compute pair_offset and rope_sign
        pair_offset = Int32(0)
        rope_sign = Float32(0.0)

        if in_first_half:
            pair_offset = elem_offset + half_rotary
            rope_sign = Float32(-1.0)
        if not in_first_half:
            pair_offset = elem_offset - half_rotary
            rope_sign = Float32(1.0)

        # Pre-compute frequencies for ALL 8 elements in this thread's vector
        # For non-interleaved: freq_idx = elem_idx % half_rotary
        # Each element has its own frequency!
        freq_reg = cute.make_rmem_tensor((8,), Float32)

        for i in cutlass.range_constexpr(8):
            elem_idx = elem_offset + i
            if elem_idx < rotary_dim:
                freq_idx = elem_idx % half_rotary
                exp_val = Float32(2.0) * Float32(freq_idx) / Float32(rotary_dim)
                base_freq = powf_approx(rope_rcp_theta, exp_val)
                freq_reg[i] = compute_llama31_freq(
                    base_freq, smooth_a, smooth_b, rope_rcp_scale
                )
            else:
                freq_reg[i] = Float32(0.0)

        if token_idx < nnz:
            pos = pos_ids[token_idx]

            # Process Q or K based on head index
            if is_q_head:
                # Get pointers for main vector and pair vector
                base_offset = (
                    token_idx * num_qo_heads + head_idx
                ) * head_dim + elem_offset
                pair_base_offset = (
                    token_idx * num_qo_heads + head_idx
                ) * head_dim + pair_offset
                q_ptr = get_ptr_as_int64(q, base_offset)
                q_pair_ptr = get_ptr_as_int64(q, pair_base_offset)
                q_rope_ptr = get_ptr_as_int64(q_rope, base_offset)

                # Vectorized load: main vector (8 elements)
                v0, v1, v2, v3 = ld_global_v4_u32(q_ptr)
                # Vectorized load: pair vector (8 elements)
                p0, p1, p2, p3 = ld_global_v4_u32(q_pair_ptr)

                # Output registers
                out_v0 = v0
                out_v1 = v1
                out_v2 = v2
                out_v3 = v3

                # Process each half2 - each element gets its own frequency!
                # Apply RoPE: y = x * cos + rope_sign * pair * sin
                # rope_sign was pre-computed above (-1 if first half, +1 if second half)

                # Elements (0,1)
                elem_idx = elem_offset
                if elem_idx < rotary_dim:
                    # Element 0
                    embed0 = Float32(pos) * freq_reg[0]
                    sin0, cos0 = sincos_approx(embed0)
                    # Element 1
                    embed1 = Float32(pos) * freq_reg[1]
                    sin1, cos1 = sincos_approx(embed1)

                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v0)
                        px0, px1 = half2_to_float2(p0)
                    else:
                        x0, x1 = bfloat2_to_float2(v0)
                        px0, px1 = bfloat2_to_float2(p0)

                    y0 = x0 * cos0 + rope_sign * px0 * sin0
                    y1 = x1 * cos1 + rope_sign * px1 * sin1

                    if cutlass.const_expr(is_fp16):
                        out_v0 = float2_to_half2(y0, y1)
                    else:
                        out_v0 = float2_to_bfloat2(y0, y1)

                # Elements (2,3)
                elem_idx = elem_offset + 2
                if elem_idx < rotary_dim:
                    embed0 = Float32(pos) * freq_reg[2]
                    sin0, cos0 = sincos_approx(embed0)
                    embed1 = Float32(pos) * freq_reg[3]
                    sin1, cos1 = sincos_approx(embed1)

                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v1)
                        px0, px1 = half2_to_float2(p1)
                    else:
                        x0, x1 = bfloat2_to_float2(v1)
                        px0, px1 = bfloat2_to_float2(p1)

                    y0 = x0 * cos0 + rope_sign * px0 * sin0
                    y1 = x1 * cos1 + rope_sign * px1 * sin1

                    if cutlass.const_expr(is_fp16):
                        out_v1 = float2_to_half2(y0, y1)
                    else:
                        out_v1 = float2_to_bfloat2(y0, y1)

                # Elements (4,5)
                elem_idx = elem_offset + 4
                if elem_idx < rotary_dim:
                    embed0 = Float32(pos) * freq_reg[4]
                    sin0, cos0 = sincos_approx(embed0)
                    embed1 = Float32(pos) * freq_reg[5]
                    sin1, cos1 = sincos_approx(embed1)

                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v2)
                        px0, px1 = half2_to_float2(p2)
                    else:
                        x0, x1 = bfloat2_to_float2(v2)
                        px0, px1 = bfloat2_to_float2(p2)

                    y0 = x0 * cos0 + rope_sign * px0 * sin0
                    y1 = x1 * cos1 + rope_sign * px1 * sin1

                    if cutlass.const_expr(is_fp16):
                        out_v2 = float2_to_half2(y0, y1)
                    else:
                        out_v2 = float2_to_bfloat2(y0, y1)

                # Elements (6,7)
                elem_idx = elem_offset + 6
                if elem_idx < rotary_dim:
                    embed0 = Float32(pos) * freq_reg[6]
                    sin0, cos0 = sincos_approx(embed0)
                    embed1 = Float32(pos) * freq_reg[7]
                    sin1, cos1 = sincos_approx(embed1)

                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v3)
                        px0, px1 = half2_to_float2(p3)
                    else:
                        x0, x1 = bfloat2_to_float2(v3)
                        px0, px1 = bfloat2_to_float2(p3)

                    y0 = x0 * cos0 + rope_sign * px0 * sin0
                    y1 = x1 * cos1 + rope_sign * px1 * sin1

                    if cutlass.const_expr(is_fp16):
                        out_v3 = float2_to_half2(y0, y1)
                    else:
                        out_v3 = float2_to_bfloat2(y0, y1)

                # Vectorized store
                st_global_v4_u32(q_rope_ptr, out_v0, out_v1, out_v2, out_v3)

            # Process K
            if not is_q_head:
                base_offset = (
                    token_idx * num_kv_heads + k_head_idx
                ) * head_dim + elem_offset
                pair_base_offset = (
                    token_idx * num_kv_heads + k_head_idx
                ) * head_dim + pair_offset
                k_ptr = get_ptr_as_int64(k, base_offset)
                k_pair_ptr = get_ptr_as_int64(k, pair_base_offset)
                k_rope_ptr = get_ptr_as_int64(k_rope, base_offset)

                v0, v1, v2, v3 = ld_global_v4_u32(k_ptr)
                p0, p1, p2, p3 = ld_global_v4_u32(k_pair_ptr)

                out_v0 = v0
                out_v1 = v1
                out_v2 = v2
                out_v3 = v3

                # Elements (0,1)
                elem_idx = elem_offset
                if elem_idx < rotary_dim:
                    embed0 = Float32(pos) * freq_reg[0]
                    sin0, cos0 = sincos_approx(embed0)
                    embed1 = Float32(pos) * freq_reg[1]
                    sin1, cos1 = sincos_approx(embed1)

                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v0)
                        px0, px1 = half2_to_float2(p0)
                    else:
                        x0, x1 = bfloat2_to_float2(v0)
                        px0, px1 = bfloat2_to_float2(p0)

                    y0 = x0 * cos0 + rope_sign * px0 * sin0
                    y1 = x1 * cos1 + rope_sign * px1 * sin1

                    if cutlass.const_expr(is_fp16):
                        out_v0 = float2_to_half2(y0, y1)
                    else:
                        out_v0 = float2_to_bfloat2(y0, y1)

                # Elements (2,3)
                elem_idx = elem_offset + 2
                if elem_idx < rotary_dim:
                    embed0 = Float32(pos) * freq_reg[2]
                    sin0, cos0 = sincos_approx(embed0)
                    embed1 = Float32(pos) * freq_reg[3]
                    sin1, cos1 = sincos_approx(embed1)

                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v1)
                        px0, px1 = half2_to_float2(p1)
                    else:
                        x0, x1 = bfloat2_to_float2(v1)
                        px0, px1 = bfloat2_to_float2(p1)

                    y0 = x0 * cos0 + rope_sign * px0 * sin0
                    y1 = x1 * cos1 + rope_sign * px1 * sin1

                    if cutlass.const_expr(is_fp16):
                        out_v1 = float2_to_half2(y0, y1)
                    else:
                        out_v1 = float2_to_bfloat2(y0, y1)

                # Elements (4,5)
                elem_idx = elem_offset + 4
                if elem_idx < rotary_dim:
                    embed0 = Float32(pos) * freq_reg[4]
                    sin0, cos0 = sincos_approx(embed0)
                    embed1 = Float32(pos) * freq_reg[5]
                    sin1, cos1 = sincos_approx(embed1)

                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v2)
                        px0, px1 = half2_to_float2(p2)
                    else:
                        x0, x1 = bfloat2_to_float2(v2)
                        px0, px1 = bfloat2_to_float2(p2)

                    y0 = x0 * cos0 + rope_sign * px0 * sin0
                    y1 = x1 * cos1 + rope_sign * px1 * sin1

                    if cutlass.const_expr(is_fp16):
                        out_v2 = float2_to_half2(y0, y1)
                    else:
                        out_v2 = float2_to_bfloat2(y0, y1)

                # Elements (6,7)
                elem_idx = elem_offset + 6
                if elem_idx < rotary_dim:
                    embed0 = Float32(pos) * freq_reg[6]
                    sin0, cos0 = sincos_approx(embed0)
                    embed1 = Float32(pos) * freq_reg[7]
                    sin1, cos1 = sincos_approx(embed1)

                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v3)
                        px0, px1 = half2_to_float2(p3)
                    else:
                        x0, x1 = bfloat2_to_float2(v3)
                        px0, px1 = bfloat2_to_float2(p3)

                    y0 = x0 * cos0 + rope_sign * px0 * sin0
                    y1 = x1 * cos1 + rope_sign * px1 * sin1

                    if cutlass.const_expr(is_fp16):
                        out_v3 = float2_to_half2(y0, y1)
                    else:
                        out_v3 = float2_to_bfloat2(y0, y1)

                st_global_v4_u32(k_rope_ptr, out_v0, out_v1, out_v2, out_v3)


class RopeKernelInterleavedVec:
    """
    Vectorized RoPE kernel (interleaved/GPT-J style).

    Performance optimizations:
    1. Uses 128-bit vectorized loads/stores (8 fp16 elements at a time)
    2. Processes half2 pairs directly (perfect for interleaved mode where pairs are adjacent)
    3. Pre-computes frequencies once per thread
    4. Uses sincos_approx for fused sin/cos computation

    For interleaved mode, pairs are adjacent: (e0,e1), (e2,e3), ...
    Each half2/bfloat2 contains exactly one RoPE pair, which is ideal for vectorization.

    Thread block configuration:
    - Each thread handles 8 elements (128 bits = 4 half2 pairs)
    - bdx = head_dim / 8 threads per head
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        head_dim: int,
        rotary_dim: int,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.half_rotary = rotary_dim // 2

        # Vectorized: each thread handles 8 elements (4 half2 pairs)
        # 128 bits = 4 x u32 = 8 x fp16
        self.elems_per_thread = 8
        self.bdx = head_dim // self.elems_per_thread
        self.num_threads = max(128, self.bdx)
        self.bdy = self.num_threads // self.bdx

        # Determine if fp16 or bf16
        self.is_fp16 = dtype.width == 16 and dtype == cutlass.Float16

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        q_rope: cute.Tensor,
        k_rope: cute.Tensor,
        pos_ids: cute.Tensor,
        nnz: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
        rope_rcp_scale: Float32,
        rope_rcp_theta: Float32,
        smooth_a: Float32,
        smooth_b: Float32,
        stream,
    ):
        tokens_per_block = self.bdy
        num_token_blocks = (nnz + tokens_per_block - 1) // tokens_per_block
        total_heads = num_qo_heads + num_kv_heads

        self.kernel(
            q,
            k,
            q_rope,
            k_rope,
            pos_ids,
            nnz,
            num_qo_heads,
            num_kv_heads,
            rope_rcp_scale,
            rope_rcp_theta,
            smooth_a,
            smooth_b,
        ).launch(
            grid=[num_token_blocks, total_heads, 1],
            block=[self.bdx, self.bdy, 1],
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        q_rope: cute.Tensor,
        k_rope: cute.Tensor,
        pos_ids: cute.Tensor,
        nnz: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
        rope_rcp_scale: Float32,
        rope_rcp_theta: Float32,
        smooth_a: Float32,
        smooth_b: Float32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        head_dim = self.head_dim
        rotary_dim = self.rotary_dim
        elems_per_thread = self.elems_per_thread
        bdy = self.bdy
        is_fp16 = self.is_fp16

        token_idx = bidx * bdy + tidy
        elem_offset = tidx * elems_per_thread

        # Determine if this block handles Q or K
        is_q_head = bidy < num_qo_heads
        head_idx = bidy
        k_head_idx = bidy - num_qo_heads

        # Pre-compute frequencies for this thread's 4 pairs
        # Each pair uses frequency at index (elem_offset + 2*i) / 2 = elem_offset/2 + i
        freq_reg = cute.make_rmem_tensor((4,), Float32)

        for i in cutlass.range_constexpr(4):
            # For interleaved: freq_idx = pair_index = elem_offset/2 + i
            pair_start = elem_offset + i * 2  # Start element of pair
            if pair_start < rotary_dim:
                freq_idx = pair_start // 2
                exp_val = Float32(2.0) * Float32(freq_idx) / Float32(rotary_dim)
                base_freq = powf_approx(rope_rcp_theta, exp_val)
                freq_reg[i] = compute_llama31_freq(
                    base_freq, smooth_a, smooth_b, rope_rcp_scale
                )
            else:
                freq_reg[i] = Float32(0.0)

        if token_idx < nnz:
            pos = pos_ids[token_idx]

            # Process Q or K based on head index
            if is_q_head:
                # Get pointer to Q data for this token and head
                base_offset = (
                    token_idx * num_qo_heads + head_idx
                ) * head_dim + elem_offset
                q_ptr = get_ptr_as_int64(q, base_offset)
                q_rope_ptr = get_ptr_as_int64(q_rope, base_offset)

                # Vectorized load: 4 x u32 = 8 fp16 elements
                v0, v1, v2, v3 = ld_global_v4_u32(q_ptr)

                # Process each half2 pair (4 pairs total)
                out_v0 = v0
                out_v1 = v1
                out_v2 = v2
                out_v3 = v3

                # Pair 0: elements 0,1
                pair_start = elem_offset
                if pair_start < rotary_dim:
                    freq = freq_reg[0]
                    embed = Float32(pos) * freq
                    sin_val, cos_val = sincos_approx(embed)
                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v0)
                    else:
                        x0, x1 = bfloat2_to_float2(v0)
                    y0 = x0 * cos_val - x1 * sin_val
                    y1 = x0 * sin_val + x1 * cos_val
                    if cutlass.const_expr(is_fp16):
                        out_v0 = float2_to_half2(y0, y1)
                    else:
                        out_v0 = float2_to_bfloat2(y0, y1)

                # Pair 1: elements 2,3
                pair_start = elem_offset + 2
                if pair_start < rotary_dim:
                    freq = freq_reg[1]
                    embed = Float32(pos) * freq
                    sin_val, cos_val = sincos_approx(embed)
                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v1)
                    else:
                        x0, x1 = bfloat2_to_float2(v1)
                    y0 = x0 * cos_val - x1 * sin_val
                    y1 = x0 * sin_val + x1 * cos_val
                    if cutlass.const_expr(is_fp16):
                        out_v1 = float2_to_half2(y0, y1)
                    else:
                        out_v1 = float2_to_bfloat2(y0, y1)

                # Pair 2: elements 4,5
                pair_start = elem_offset + 4
                if pair_start < rotary_dim:
                    freq = freq_reg[2]
                    embed = Float32(pos) * freq
                    sin_val, cos_val = sincos_approx(embed)
                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v2)
                    else:
                        x0, x1 = bfloat2_to_float2(v2)
                    y0 = x0 * cos_val - x1 * sin_val
                    y1 = x0 * sin_val + x1 * cos_val
                    if cutlass.const_expr(is_fp16):
                        out_v2 = float2_to_half2(y0, y1)
                    else:
                        out_v2 = float2_to_bfloat2(y0, y1)

                # Pair 3: elements 6,7
                pair_start = elem_offset + 6
                if pair_start < rotary_dim:
                    freq = freq_reg[3]
                    embed = Float32(pos) * freq
                    sin_val, cos_val = sincos_approx(embed)
                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v3)
                    else:
                        x0, x1 = bfloat2_to_float2(v3)
                    y0 = x0 * cos_val - x1 * sin_val
                    y1 = x0 * sin_val + x1 * cos_val
                    if cutlass.const_expr(is_fp16):
                        out_v3 = float2_to_half2(y0, y1)
                    else:
                        out_v3 = float2_to_bfloat2(y0, y1)

                # Vectorized store
                st_global_v4_u32(q_rope_ptr, out_v0, out_v1, out_v2, out_v3)

            # Process K
            if not is_q_head:
                # Get pointer to K data for this token and head
                base_offset = (
                    token_idx * num_kv_heads + k_head_idx
                ) * head_dim + elem_offset
                k_ptr = get_ptr_as_int64(k, base_offset)
                k_rope_ptr = get_ptr_as_int64(k_rope, base_offset)

                # Vectorized load
                v0, v1, v2, v3 = ld_global_v4_u32(k_ptr)

                out_v0 = v0
                out_v1 = v1
                out_v2 = v2
                out_v3 = v3

                # Pair 0
                pair_start = elem_offset
                if pair_start < rotary_dim:
                    freq = freq_reg[0]
                    embed = Float32(pos) * freq
                    sin_val, cos_val = sincos_approx(embed)
                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v0)
                    else:
                        x0, x1 = bfloat2_to_float2(v0)
                    y0 = x0 * cos_val - x1 * sin_val
                    y1 = x0 * sin_val + x1 * cos_val
                    if cutlass.const_expr(is_fp16):
                        out_v0 = float2_to_half2(y0, y1)
                    else:
                        out_v0 = float2_to_bfloat2(y0, y1)

                # Pair 1
                pair_start = elem_offset + 2
                if pair_start < rotary_dim:
                    freq = freq_reg[1]
                    embed = Float32(pos) * freq
                    sin_val, cos_val = sincos_approx(embed)
                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v1)
                    else:
                        x0, x1 = bfloat2_to_float2(v1)
                    y0 = x0 * cos_val - x1 * sin_val
                    y1 = x0 * sin_val + x1 * cos_val
                    if cutlass.const_expr(is_fp16):
                        out_v1 = float2_to_half2(y0, y1)
                    else:
                        out_v1 = float2_to_bfloat2(y0, y1)

                # Pair 2
                pair_start = elem_offset + 4
                if pair_start < rotary_dim:
                    freq = freq_reg[2]
                    embed = Float32(pos) * freq
                    sin_val, cos_val = sincos_approx(embed)
                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v2)
                    else:
                        x0, x1 = bfloat2_to_float2(v2)
                    y0 = x0 * cos_val - x1 * sin_val
                    y1 = x0 * sin_val + x1 * cos_val
                    if cutlass.const_expr(is_fp16):
                        out_v2 = float2_to_half2(y0, y1)
                    else:
                        out_v2 = float2_to_bfloat2(y0, y1)

                # Pair 3
                pair_start = elem_offset + 6
                if pair_start < rotary_dim:
                    freq = freq_reg[3]
                    embed = Float32(pos) * freq
                    sin_val, cos_val = sincos_approx(embed)
                    if cutlass.const_expr(is_fp16):
                        x0, x1 = half2_to_float2(v3)
                    else:
                        x0, x1 = bfloat2_to_float2(v3)
                    y0 = x0 * cos_val - x1 * sin_val
                    y1 = x0 * sin_val + x1 * cos_val
                    if cutlass.const_expr(is_fp16):
                        out_v3 = float2_to_half2(y0, y1)
                    else:
                        out_v3 = float2_to_bfloat2(y0, y1)

                # Vectorized store
                st_global_v4_u32(k_rope_ptr, out_v0, out_v1, out_v2, out_v3)


# =============================================================================
# Kernel Caching
# =============================================================================


@functools.lru_cache(maxsize=64)
def _get_compiled_kernel(
    head_dim: int,
    rotary_dim: int,
    interleave: bool,
    dtype_str: str,
) -> Callable:
    """Get or compile a cached RoPE kernel.

    Uses 128-bit vectorized loads/stores for optimal performance.
    """
    dtype = get_cutlass_dtype(dtype_str)

    kernel_obj: Union[RopeKernelInterleavedVec, RopeKernelNonInterleavedVec]
    if interleave:
        # Interleaved mode: pairs are adjacent, perfect for 128-bit loads
        kernel_obj = RopeKernelInterleavedVec(
            dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim
        )
    else:
        # Non-interleaved: load main + pair vectors using 128-bit loads
        # (like CUDA C++ vec_apply_llama_rope_cos_sin)
        kernel_obj = RopeKernelNonInterleavedVec(
            dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim
        )

    sym_nnz = cute.sym_int()
    sym_num_qo_heads = cute.sym_int()
    sym_num_kv_heads = cute.sym_int()

    q_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    pos_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_nnz,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        q_rope_fake,
        k_rope_fake,
        pos_ids_fake,
        Int32(1),  # nnz
        Int32(1),  # num_qo_heads
        Int32(1),  # num_kv_heads
        Float32(1.0),  # rope_rcp_scale
        Float32(1.0),  # rope_rcp_theta
        Float32(0.0),  # smooth_a
        Float32(1.0),  # smooth_b
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        q: torch.Tensor,
        k: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        pos_ids: torch.Tensor,
        nnz: int,
        num_qo_heads: int,
        num_kv_heads: int,
        rope_rcp_scale: float,
        rope_rcp_theta: float,
        smooth_a: float,
        smooth_b: float,
    ) -> None:
        compiled_kernel(
            q,
            k,
            q_rope,
            k_rope,
            pos_ids,
            Int32(nnz),
            Int32(num_qo_heads),
            Int32(num_kv_heads),
            Float32(rope_rcp_scale),
            Float32(rope_rcp_theta),
            Float32(smooth_a),
            Float32(smooth_b),
        )

    return tensor_api


# =============================================================================
# Public API
# =============================================================================


def apply_rope_cute_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 1.0,
    old_context_len: int = 8192,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE (Rotary Positional Embeddings) using CuTe-DSL backend.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape: (nnz, num_q_heads, head_dim)
    k : torch.Tensor
        Key tensor, shape: (nnz, num_k_heads, head_dim)
    pos_ids : torch.Tensor
        Position indices, shape: (nnz,)
    rotary_dim : Optional[int]
        Dimension to apply RoPE. If None, uses full head_dim.
    interleave : bool
        If True, use interleaved (GPT-J) style. If False, use non-interleaved (NeoX) style.
    rope_scale : float
        Scaling factor for RoPE frequencies.
    rope_theta : float
        Base theta value for RoPE.
    low_freq_factor : float
        Llama 3.1 low frequency factor. Default 1.0 (no scaling).
    high_freq_factor : float
        Llama 3.1 high frequency factor. Default 1.0 (no scaling).
    old_context_len : int
        Llama 3.1 original context length. Default 8192.
    q_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated queries. If None, allocated internally.
    k_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated keys. If None, allocated internally.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors.

    Notes
    -----
    Llama 3.1 frequency scaling:
        When low_freq_factor != 1.0 or high_freq_factor != 1.0, applies smooth
        interpolation between scaled and unscaled frequencies. The formula is:
            smooth = clamp(freq * smooth_a + smooth_b, 0, 1)
            freq = (1 - smooth) * (freq / rope_scale) + smooth * freq
    """
    # Validate inputs
    assert q.ndim == 3, f"q must be 3D, got {q.ndim}D"
    assert k.ndim == 3, f"k must be 3D, got {k.ndim}D"
    assert q.size(0) == k.size(0), "q and k must have same nnz"
    assert q.size(2) == k.size(2), "q and k must have same head_dim"
    assert q.is_cuda, "q must be on CUDA"
    assert k.is_cuda, "k must be on CUDA"
    assert pos_ids.is_cuda, "pos_ids must be on CUDA"

    nnz = q.size(0)
    num_qo_heads = q.size(1)
    num_kv_heads = k.size(1)
    head_dim = q.size(2)

    if rotary_dim is None:
        rotary_dim = head_dim

    assert rotary_dim <= head_dim, (
        f"rotary_dim must be <= head_dim, got {rotary_dim} > {head_dim}"
    )
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    assert head_dim % 2 == 0, "head_dim must be even"

    # Determine dtype
    dtype = q.dtype
    assert dtype in [torch.float16, torch.bfloat16], f"Unsupported dtype: {dtype}"
    dtype_str = "float16" if dtype == torch.float16 else "bfloat16"

    # Allocate output tensors if not provided
    if q_rope is None:
        q_rope = torch.empty_like(q)
    if k_rope is None:
        k_rope = torch.empty_like(k)

    # Ensure contiguous tensors
    q = q.contiguous()
    k = k.contiguous()
    q_rope = q_rope.contiguous()
    k_rope = k_rope.contiguous()

    # Convert pos_ids to int32 if needed
    if pos_ids.dtype != torch.int32:
        pos_ids = pos_ids.to(torch.int32)
    pos_ids = pos_ids.contiguous()

    # Compute reciprocal scale and theta
    rope_rcp_scale = 1.0 / rope_scale
    rope_rcp_theta = 1.0 / rope_theta

    # Compute Llama 3.1 smooth_a and smooth_b
    # The CUDA kernel uses:
    #   smooth_a = old_context_len / (2 * pi * (high_freq_factor - low_freq_factor))
    #   smooth_b = -1.0 / ((high_freq_factor / low_freq_factor) - 1.0)
    #            = -low_freq_factor / (high_freq_factor - low_freq_factor)
    #
    # When high_freq_factor == low_freq_factor, this causes division by zero.
    # The CUDA kernel gets inf/NaN which clamps to smooth=0, applying full scaling.
    import math

    if high_freq_factor != low_freq_factor:
        smooth_a = old_context_len / (
            2.0 * math.pi * (high_freq_factor - low_freq_factor)
        )
        smooth_b = -1.0 / ((high_freq_factor / low_freq_factor) - 1.0)
    else:
        # When factors are equal, apply full scaling (smooth=0)
        smooth_a = 0.0
        smooth_b = 0.0  # This gives smooth=0, applying full scaling

    # Launch kernel
    kernel = _get_compiled_kernel(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        interleave=interleave,
        dtype_str=dtype_str,
    )
    kernel(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        nnz,
        num_qo_heads,
        num_kv_heads,
        rope_rcp_scale,
        rope_rcp_theta,
        smooth_a,
        smooth_b,
    )

    return q_rope, k_rope


def _compute_pos_ids_from_indptr_offsets(
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    nnz: int,
) -> torch.Tensor:
    """
    Compute position IDs from indptr and offsets.

    For token j in sequence i (where indptr[i] <= j < indptr[i+1]):
        pos_ids[j] = j - indptr[i] + offsets[i]

    Parameters
    ----------
    indptr : torch.Tensor
        Indptr tensor of shape (batch_size + 1,). indptr[i] is the start index
        of sequence i in the ragged tensor.
    offsets : torch.Tensor
        Offset tensor of shape (batch_size,). offsets[i] is the position offset
        for sequence i.
    nnz : int
        Total number of tokens (= indptr[-1]).

    Returns
    -------
    torch.Tensor
        Position IDs tensor of shape (nnz,).
    """
    device = indptr.device
    batch_size = indptr.size(0) - 1

    # Create output tensor
    pos_ids = torch.empty(nnz, dtype=torch.int32, device=device)

    # For each sequence, compute positions
    # This is a simple CPU loop - could be optimized with a Triton kernel if needed
    for i in range(batch_size):
        start = indptr[i].item()
        end = indptr[i + 1].item()
        offset = offsets[i].item()
        seq_len = end - start
        if seq_len > 0:
            pos_ids[start:end] = (
                torch.arange(seq_len, dtype=torch.int32, device=device) + offset
            )

    return pos_ids


def apply_rope_with_indptr_cute_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 1.0,
    old_context_len: int = 8192,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE using CuTe-DSL backend with indptr/offsets (ragged tensor format).

    This API is compatible with flashinfer.apply_rope.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: (nnz, num_q_heads, head_dim)
    k : torch.Tensor
        Key ragged tensor, shape: (nnz, num_k_heads, head_dim)
    indptr : torch.Tensor
        Indptr tensor, shape: (batch_size + 1,). Defines sequence boundaries.
    offsets : torch.Tensor
        Position offsets for each sequence, shape: (batch_size,).
    rotary_dim : Optional[int]
        Dimension to apply RoPE. If None, uses full head_dim.
    interleave : bool
        If True, use interleaved (GPT-J) style. If False, use non-interleaved (NeoX) style.
    rope_scale : float
        Scaling factor for RoPE frequencies.
    rope_theta : float
        Base theta value for RoPE.
    low_freq_factor : float
        Llama 3.1 low frequency factor. Default 1.0 (no scaling).
    high_freq_factor : float
        Llama 3.1 high frequency factor. Default 1.0 (no scaling).
    old_context_len : int
        Llama 3.1 original context length. Default 8192.
    q_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated queries. If None, allocated internally.
    k_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated keys. If None, allocated internally.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors.
    """
    nnz = q.size(0)

    # Compute pos_ids from indptr and offsets
    pos_ids = _compute_pos_ids_from_indptr_offsets(indptr, offsets, nnz)

    # Delegate to the pos_ids-based API
    return apply_rope_cute_dsl(
        q=q,
        k=k,
        pos_ids=pos_ids,
        rotary_dim=rotary_dim,
        interleave=interleave,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        old_context_len=old_context_len,
        q_rope=q_rope,
        k_rope=k_rope,
    )


def apply_llama31_rope_with_indptr_cute_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8.0,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Llama 3.1 style RoPE using CuTe-DSL backend with indptr/offsets.

    This API is compatible with flashinfer.apply_llama31_rope.

    Parameters are the same as apply_rope_with_indptr_cute_dsl, but with
    Llama 3.1 default parameters.
    """
    return apply_rope_with_indptr_cute_dsl(
        q=q,
        k=k,
        indptr=indptr,
        offsets=offsets,
        rotary_dim=rotary_dim,
        interleave=interleave,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        old_context_len=old_context_len,
        q_rope=q_rope,
        k_rope=k_rope,
    )


__all__ = [
    "RopeKernelNonInterleavedVec",
    "RopeKernelInterleavedVec",
]
