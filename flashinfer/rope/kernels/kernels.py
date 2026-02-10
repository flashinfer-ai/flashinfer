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

CuTe-DSL RoPE Kernel Classes
============================

This module contains the kernel classes for RoPE computation:
- RopeKernelNonInterleavedVec: Vectorized kernel for non-interleaved (NeoX) style
- RopeKernelInterleavedVec: Vectorized kernel for interleaved (GPT-J) style
- RopeKernelSeqHeads: Sequential-head kernel for large workloads
- RopeKernelWithIndptr: Kernel accepting indptr/offsets directly
- RopeKernelCosSinCache: Kernel using precomputed cos/sin cache
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from .ptx_ops import (
    get_ptr_as_int64,
    ld_global_v4_f32,
    ld_global_v4_u32,
    st_global_v4_u32,
    sincos_approx,
    powf_approx,
    half2_to_float2,
    bfloat2_to_float2,
    float2_to_half2,
    float2_to_bfloat2,
)
from .helpers import (
    apply_rope_interleaved_fp16,
    apply_rope_interleaved_bf16,
    apply_rope_non_interleaved_fp16,
    apply_rope_non_interleaved_bf16,
    compute_llama31_freq,
)


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

    Note: This kernel parallelizes over heads (one block per head).
    For large workloads, use RopeKernelSeqHeads instead.
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

            # Pre-compute sin/cos for this position
            sin_reg = cute.make_rmem_tensor((4,), Float32)
            cos_reg = cute.make_rmem_tensor((4,), Float32)

            for i in cutlass.range_constexpr(4):
                pair_start = elem_offset + i * 2
                if pair_start < rotary_dim:
                    embed = Float32(pos) * freq_reg[i]
                    sin_reg[i], cos_reg[i] = sincos_approx(embed)

            # Process Q or K based on head index
            if is_q_head:
                base_offset = (
                    token_idx * num_qo_heads + head_idx
                ) * head_dim + elem_offset
                q_ptr = get_ptr_as_int64(q, base_offset)
                q_rope_ptr = get_ptr_as_int64(q_rope, base_offset)

                v0, v1, v2, v3 = ld_global_v4_u32(q_ptr)
                out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                # Process each pair directly with helpers
                if elem_offset < rotary_dim:
                    if cutlass.const_expr(is_fp16):
                        out_v0 = apply_rope_interleaved_fp16(v0, sin_reg[0], cos_reg[0])
                    else:
                        out_v0 = apply_rope_interleaved_bf16(v0, sin_reg[0], cos_reg[0])
                if elem_offset + 2 < rotary_dim:
                    if cutlass.const_expr(is_fp16):
                        out_v1 = apply_rope_interleaved_fp16(v1, sin_reg[1], cos_reg[1])
                    else:
                        out_v1 = apply_rope_interleaved_bf16(v1, sin_reg[1], cos_reg[1])
                if elem_offset + 4 < rotary_dim:
                    if cutlass.const_expr(is_fp16):
                        out_v2 = apply_rope_interleaved_fp16(v2, sin_reg[2], cos_reg[2])
                    else:
                        out_v2 = apply_rope_interleaved_bf16(v2, sin_reg[2], cos_reg[2])
                if elem_offset + 6 < rotary_dim:
                    if cutlass.const_expr(is_fp16):
                        out_v3 = apply_rope_interleaved_fp16(v3, sin_reg[3], cos_reg[3])
                    else:
                        out_v3 = apply_rope_interleaved_bf16(v3, sin_reg[3], cos_reg[3])

                st_global_v4_u32(q_rope_ptr, out_v0, out_v1, out_v2, out_v3)

            # Process K
            if not is_q_head:
                base_offset = (
                    token_idx * num_kv_heads + k_head_idx
                ) * head_dim + elem_offset
                k_ptr = get_ptr_as_int64(k, base_offset)
                k_rope_ptr = get_ptr_as_int64(k_rope, base_offset)

                v0, v1, v2, v3 = ld_global_v4_u32(k_ptr)
                out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                if elem_offset < rotary_dim:
                    if cutlass.const_expr(is_fp16):
                        out_v0 = apply_rope_interleaved_fp16(v0, sin_reg[0], cos_reg[0])
                    else:
                        out_v0 = apply_rope_interleaved_bf16(v0, sin_reg[0], cos_reg[0])
                if elem_offset + 2 < rotary_dim:
                    if cutlass.const_expr(is_fp16):
                        out_v1 = apply_rope_interleaved_fp16(v1, sin_reg[1], cos_reg[1])
                    else:
                        out_v1 = apply_rope_interleaved_bf16(v1, sin_reg[1], cos_reg[1])
                if elem_offset + 4 < rotary_dim:
                    if cutlass.const_expr(is_fp16):
                        out_v2 = apply_rope_interleaved_fp16(v2, sin_reg[2], cos_reg[2])
                    else:
                        out_v2 = apply_rope_interleaved_bf16(v2, sin_reg[2], cos_reg[2])
                if elem_offset + 6 < rotary_dim:
                    if cutlass.const_expr(is_fp16):
                        out_v3 = apply_rope_interleaved_fp16(v3, sin_reg[3], cos_reg[3])
                    else:
                        out_v3 = apply_rope_interleaved_bf16(v3, sin_reg[3], cos_reg[3])

                st_global_v4_u32(k_rope_ptr, out_v0, out_v1, out_v2, out_v3)


class RopeKernelSeqHeads:
    """
    Sequential-head RoPE kernel for large workloads.

    This kernel matches CUDA's BatchQKApplyRotaryPosIdsKernel structure:
    - Grid: [num_token_blocks] (not multiplied by heads!)
    - Each thread block loops over ALL heads sequentially
    - Sin/cos computed ONCE per token, reused for all heads

    This is more efficient at large seq_len because:
    1. Fewer blocks = less launch overhead
    2. Sin/cos reuse saves compute (40 heads reuse same sin/cos)

    Used when: num_token_blocks >= GPU occupancy threshold
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        head_dim: int,
        rotary_dim: int,
        interleave: bool,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.half_rotary = rotary_dim // 2
        self.interleave = interleave

        self.elems_per_thread = 8
        self.bdx = head_dim // self.elems_per_thread
        self.num_threads = max(128, self.bdx)
        self.bdy = self.num_threads // self.bdx

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

        # Key difference: grid.y = 1, not total_heads!
        # Each block loops over all heads
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
            grid=[num_token_blocks, 1, 1],
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
        bidx, _, _ = cute.arch.block_idx()

        head_dim = self.head_dim
        rotary_dim = self.rotary_dim
        half_rotary = self.half_rotary
        elems_per_thread = self.elems_per_thread
        bdy = self.bdy
        is_fp16 = self.is_fp16
        interleave = self.interleave

        token_idx = bidx * bdy + tidy
        elem_offset = tidx * elems_per_thread

        # For non-interleaved: pre-compute pair offset and sign
        pair_elem_offset = Int32(0)
        rope_sign = Float32(0.0)
        if not interleave:
            in_first_half = elem_offset < half_rotary
            if in_first_half:
                pair_elem_offset = elem_offset + half_rotary
                rope_sign = Float32(-1.0)
            if not in_first_half:
                pair_elem_offset = elem_offset - half_rotary
                rope_sign = Float32(1.0)

        # Pre-compute frequencies (position-independent)
        freq_reg = cute.make_rmem_tensor((8,), Float32)
        if interleave:
            for i in cutlass.range_constexpr(4):
                pair_start = elem_offset + i * 2
                if pair_start < rotary_dim:
                    freq_idx = pair_start // 2
                    exp_val = Float32(2.0) * Float32(freq_idx) / Float32(rotary_dim)
                    base_freq = powf_approx(rope_rcp_theta, exp_val)
                    freq_reg[i] = compute_llama31_freq(
                        base_freq, smooth_a, smooth_b, rope_rcp_scale
                    )
                else:
                    freq_reg[i] = Float32(0.0)
        else:
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

            # Pre-compute sin/cos ONCE for this token (reused for all heads!)
            sin_reg = cute.make_rmem_tensor((8,), Float32)
            cos_reg = cute.make_rmem_tensor((8,), Float32)

            if interleave:
                for i in cutlass.range_constexpr(4):
                    pair_start = elem_offset + i * 2
                    if pair_start < rotary_dim:
                        embed = Float32(pos) * freq_reg[i]
                        sin_reg[i], cos_reg[i] = sincos_approx(embed)
            else:
                for i in cutlass.range_constexpr(8):
                    elem_idx = elem_offset + i
                    if elem_idx < rotary_dim:
                        embed = Float32(pos) * freq_reg[i]
                        sin_reg[i], cos_reg[i] = sincos_approx(embed)

            # Loop over ALL Q heads (reuses sin/cos!)
            for qo_head_idx in range(num_qo_heads):
                base_offset = (
                    token_idx * num_qo_heads + qo_head_idx
                ) * head_dim + elem_offset
                q_ptr = get_ptr_as_int64(q, base_offset)
                q_rope_ptr = get_ptr_as_int64(q_rope, base_offset)

                v0, v1, v2, v3 = ld_global_v4_u32(q_ptr)
                out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                if interleave:
                    # Interleaved: process each pair directly with helper functions
                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_interleaved_fp16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                        else:
                            out_v0 = apply_rope_interleaved_bf16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_interleaved_fp16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                        else:
                            out_v1 = apply_rope_interleaved_bf16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_interleaved_fp16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                        else:
                            out_v2 = apply_rope_interleaved_bf16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_interleaved_fp16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                        else:
                            out_v3 = apply_rope_interleaved_bf16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                else:
                    # Non-interleaved: load pair vector
                    pair_base_offset = (
                        token_idx * num_qo_heads + qo_head_idx
                    ) * head_dim + pair_elem_offset
                    q_pair_ptr = get_ptr_as_int64(q, pair_base_offset)
                    p0, p1, p2, p3 = ld_global_v4_u32(q_pair_ptr)

                    # Process each pair directly with helper functions
                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_non_interleaved_fp16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                        else:
                            out_v0 = apply_rope_non_interleaved_bf16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_non_interleaved_fp16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                        else:
                            out_v1 = apply_rope_non_interleaved_bf16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_non_interleaved_fp16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                        else:
                            out_v2 = apply_rope_non_interleaved_bf16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_non_interleaved_fp16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )
                        else:
                            out_v3 = apply_rope_non_interleaved_bf16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )

                st_global_v4_u32(q_rope_ptr, out_v0, out_v1, out_v2, out_v3)

            # Loop over ALL K heads (also reuses sin/cos!)
            for kv_head_idx in range(num_kv_heads):
                base_offset = (
                    token_idx * num_kv_heads + kv_head_idx
                ) * head_dim + elem_offset
                k_ptr = get_ptr_as_int64(k, base_offset)
                k_rope_ptr = get_ptr_as_int64(k_rope, base_offset)

                v0, v1, v2, v3 = ld_global_v4_u32(k_ptr)
                out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                if interleave:
                    # Interleaved: process each pair directly
                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_interleaved_fp16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                        else:
                            out_v0 = apply_rope_interleaved_bf16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_interleaved_fp16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                        else:
                            out_v1 = apply_rope_interleaved_bf16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_interleaved_fp16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                        else:
                            out_v2 = apply_rope_interleaved_bf16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_interleaved_fp16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                        else:
                            out_v3 = apply_rope_interleaved_bf16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                else:
                    # Non-interleaved: load pair vector
                    pair_base_offset = (
                        token_idx * num_kv_heads + kv_head_idx
                    ) * head_dim + pair_elem_offset
                    k_pair_ptr = get_ptr_as_int64(k, pair_base_offset)
                    p0, p1, p2, p3 = ld_global_v4_u32(k_pair_ptr)

                    # Process each pair directly
                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_non_interleaved_fp16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                        else:
                            out_v0 = apply_rope_non_interleaved_bf16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_non_interleaved_fp16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                        else:
                            out_v1 = apply_rope_non_interleaved_bf16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_non_interleaved_fp16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                        else:
                            out_v2 = apply_rope_non_interleaved_bf16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_non_interleaved_fp16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )
                        else:
                            out_v3 = apply_rope_non_interleaved_bf16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )

                st_global_v4_u32(k_rope_ptr, out_v0, out_v1, out_v2, out_v3)


class RopeKernelWithIndptr:
    """
    RoPE kernel that accepts indptr/offsets directly (matches CUDA kernel structure).

    Grid structure: grid.x = batch_size, grid.y = num_heads
    Each block handles one batch and loops over tokens within that batch.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        head_dim: int,
        rotary_dim: int,
        interleave: bool,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.half_rotary = rotary_dim // 2
        self.interleave = interleave

        self.elems_per_thread = 8
        self.bdx = head_dim // self.elems_per_thread
        self.num_threads = max(128, self.bdx)
        self.bdy = self.num_threads // self.bdx

        self.is_fp16 = dtype.width == 16 and dtype == cutlass.Float16

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        q_rope: cute.Tensor,
        k_rope: cute.Tensor,
        indptr: cute.Tensor,
        offsets: cute.Tensor,
        batch_size: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
        rope_rcp_scale: Float32,
        rope_rcp_theta: Float32,
        smooth_a: Float32,
        smooth_b: Float32,
        stream,
    ):
        total_heads = num_qo_heads + num_kv_heads

        self.kernel(
            q,
            k,
            q_rope,
            k_rope,
            indptr,
            offsets,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            rope_rcp_scale,
            rope_rcp_theta,
            smooth_a,
            smooth_b,
        ).launch(
            grid=[batch_size, total_heads, 1],
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
        indptr: cute.Tensor,
        offsets: cute.Tensor,
        batch_size: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
        rope_rcp_scale: Float32,
        rope_rcp_theta: Float32,
        smooth_a: Float32,
        smooth_b: Float32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        batch_idx, head_idx, _ = cute.arch.block_idx()

        head_dim = self.head_dim
        rotary_dim = self.rotary_dim
        half_rotary = self.half_rotary
        elems_per_thread = self.elems_per_thread
        bdy = self.bdy
        is_fp16 = self.is_fp16
        interleave = self.interleave

        elem_offset = tidx * elems_per_thread

        seq_start = indptr[batch_idx]
        seq_end = indptr[batch_idx + Int32(1)]
        seq_len = seq_end - seq_start
        pos_offset = offsets[batch_idx]

        is_q_head = head_idx < num_qo_heads
        q_head_idx = head_idx
        k_head_idx = head_idx - num_qo_heads

        # For non-interleaved: pre-compute pair offset and sign
        pair_elem_offset = Int32(0)
        rope_sign = Float32(0.0)
        if not interleave:
            in_first_half = elem_offset < half_rotary
            if in_first_half:
                pair_elem_offset = elem_offset + half_rotary
                rope_sign = Float32(-1.0)
            if not in_first_half:
                pair_elem_offset = elem_offset - half_rotary
                rope_sign = Float32(1.0)

        # Pre-compute frequencies
        freq_reg = cute.make_rmem_tensor((8,), Float32)
        if interleave:
            for i in cutlass.range_constexpr(4):
                pair_start = elem_offset + i * 2
                if pair_start < rotary_dim:
                    freq_idx = pair_start // 2
                    exp_val = Float32(2.0) * Float32(freq_idx) / Float32(rotary_dim)
                    base_freq = powf_approx(rope_rcp_theta, exp_val)
                    freq_reg[i] = compute_llama31_freq(
                        base_freq, smooth_a, smooth_b, rope_rcp_scale
                    )
                else:
                    freq_reg[i] = Float32(0.0)
        else:
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

        # Loop over tokens in this batch
        # Cast to Int32 for CuTe-DSL for loop (supports Int64 indptr/offsets)
        num_iters = Int32((seq_len + bdy - Int32(1)) // bdy)
        for iter_idx in range(num_iters):
            local_token_idx = iter_idx * bdy + tidy
            if local_token_idx < seq_len:
                token_idx = seq_start + local_token_idx
                pos = pos_offset + local_token_idx

                # Pre-compute sin/cos for this position
                sin_reg = cute.make_rmem_tensor((8,), Float32)
                cos_reg = cute.make_rmem_tensor((8,), Float32)

                if interleave:
                    for i in cutlass.range_constexpr(4):
                        pair_start = elem_offset + i * 2
                        if pair_start < rotary_dim:
                            embed = Float32(pos) * freq_reg[i]
                            sin_reg[i], cos_reg[i] = sincos_approx(embed)
                else:
                    for i in cutlass.range_constexpr(8):
                        elem_idx = elem_offset + i
                        if elem_idx < rotary_dim:
                            embed = Float32(pos) * freq_reg[i]
                            sin_reg[i], cos_reg[i] = sincos_approx(embed)

                # Process Q head
                if is_q_head:
                    base_offset = (
                        token_idx * num_qo_heads + q_head_idx
                    ) * head_dim + elem_offset
                    q_ptr = get_ptr_as_int64(q, base_offset)
                    q_rope_ptr = get_ptr_as_int64(q_rope, base_offset)

                    v0, v1, v2, v3 = ld_global_v4_u32(q_ptr)
                    out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                    if interleave:
                        if elem_offset < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v0 = apply_rope_interleaved_fp16(
                                    v0, sin_reg[0], cos_reg[0]
                                )
                            else:
                                out_v0 = apply_rope_interleaved_bf16(
                                    v0, sin_reg[0], cos_reg[0]
                                )
                        if elem_offset + 2 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v1 = apply_rope_interleaved_fp16(
                                    v1, sin_reg[1], cos_reg[1]
                                )
                            else:
                                out_v1 = apply_rope_interleaved_bf16(
                                    v1, sin_reg[1], cos_reg[1]
                                )
                        if elem_offset + 4 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v2 = apply_rope_interleaved_fp16(
                                    v2, sin_reg[2], cos_reg[2]
                                )
                            else:
                                out_v2 = apply_rope_interleaved_bf16(
                                    v2, sin_reg[2], cos_reg[2]
                                )
                        if elem_offset + 6 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v3 = apply_rope_interleaved_fp16(
                                    v3, sin_reg[3], cos_reg[3]
                                )
                            else:
                                out_v3 = apply_rope_interleaved_bf16(
                                    v3, sin_reg[3], cos_reg[3]
                                )
                    else:
                        pair_base_offset = (
                            token_idx * num_qo_heads + q_head_idx
                        ) * head_dim + pair_elem_offset
                        q_pair_ptr = get_ptr_as_int64(q, pair_base_offset)
                        p0, p1, p2, p3 = ld_global_v4_u32(q_pair_ptr)

                        if elem_offset < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v0 = apply_rope_non_interleaved_fp16(
                                    v0,
                                    p0,
                                    sin_reg[0],
                                    cos_reg[0],
                                    sin_reg[1],
                                    cos_reg[1],
                                    rope_sign,
                                )
                            else:
                                out_v0 = apply_rope_non_interleaved_bf16(
                                    v0,
                                    p0,
                                    sin_reg[0],
                                    cos_reg[0],
                                    sin_reg[1],
                                    cos_reg[1],
                                    rope_sign,
                                )
                        if elem_offset + 2 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v1 = apply_rope_non_interleaved_fp16(
                                    v1,
                                    p1,
                                    sin_reg[2],
                                    cos_reg[2],
                                    sin_reg[3],
                                    cos_reg[3],
                                    rope_sign,
                                )
                            else:
                                out_v1 = apply_rope_non_interleaved_bf16(
                                    v1,
                                    p1,
                                    sin_reg[2],
                                    cos_reg[2],
                                    sin_reg[3],
                                    cos_reg[3],
                                    rope_sign,
                                )
                        if elem_offset + 4 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v2 = apply_rope_non_interleaved_fp16(
                                    v2,
                                    p2,
                                    sin_reg[4],
                                    cos_reg[4],
                                    sin_reg[5],
                                    cos_reg[5],
                                    rope_sign,
                                )
                            else:
                                out_v2 = apply_rope_non_interleaved_bf16(
                                    v2,
                                    p2,
                                    sin_reg[4],
                                    cos_reg[4],
                                    sin_reg[5],
                                    cos_reg[5],
                                    rope_sign,
                                )
                        if elem_offset + 6 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v3 = apply_rope_non_interleaved_fp16(
                                    v3,
                                    p3,
                                    sin_reg[6],
                                    cos_reg[6],
                                    sin_reg[7],
                                    cos_reg[7],
                                    rope_sign,
                                )
                            else:
                                out_v3 = apply_rope_non_interleaved_bf16(
                                    v3,
                                    p3,
                                    sin_reg[6],
                                    cos_reg[6],
                                    sin_reg[7],
                                    cos_reg[7],
                                    rope_sign,
                                )

                    st_global_v4_u32(q_rope_ptr, out_v0, out_v1, out_v2, out_v3)

                # Process K head
                if not is_q_head:
                    base_offset = (
                        token_idx * num_kv_heads + k_head_idx
                    ) * head_dim + elem_offset
                    k_ptr = get_ptr_as_int64(k, base_offset)
                    k_rope_ptr = get_ptr_as_int64(k_rope, base_offset)

                    v0, v1, v2, v3 = ld_global_v4_u32(k_ptr)
                    out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                    if interleave:
                        if elem_offset < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v0 = apply_rope_interleaved_fp16(
                                    v0, sin_reg[0], cos_reg[0]
                                )
                            else:
                                out_v0 = apply_rope_interleaved_bf16(
                                    v0, sin_reg[0], cos_reg[0]
                                )
                        if elem_offset + 2 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v1 = apply_rope_interleaved_fp16(
                                    v1, sin_reg[1], cos_reg[1]
                                )
                            else:
                                out_v1 = apply_rope_interleaved_bf16(
                                    v1, sin_reg[1], cos_reg[1]
                                )
                        if elem_offset + 4 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v2 = apply_rope_interleaved_fp16(
                                    v2, sin_reg[2], cos_reg[2]
                                )
                            else:
                                out_v2 = apply_rope_interleaved_bf16(
                                    v2, sin_reg[2], cos_reg[2]
                                )
                        if elem_offset + 6 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v3 = apply_rope_interleaved_fp16(
                                    v3, sin_reg[3], cos_reg[3]
                                )
                            else:
                                out_v3 = apply_rope_interleaved_bf16(
                                    v3, sin_reg[3], cos_reg[3]
                                )
                    else:
                        pair_base_offset = (
                            token_idx * num_kv_heads + k_head_idx
                        ) * head_dim + pair_elem_offset
                        k_pair_ptr = get_ptr_as_int64(k, pair_base_offset)
                        p0, p1, p2, p3 = ld_global_v4_u32(k_pair_ptr)

                        if elem_offset < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v0 = apply_rope_non_interleaved_fp16(
                                    v0,
                                    p0,
                                    sin_reg[0],
                                    cos_reg[0],
                                    sin_reg[1],
                                    cos_reg[1],
                                    rope_sign,
                                )
                            else:
                                out_v0 = apply_rope_non_interleaved_bf16(
                                    v0,
                                    p0,
                                    sin_reg[0],
                                    cos_reg[0],
                                    sin_reg[1],
                                    cos_reg[1],
                                    rope_sign,
                                )
                        if elem_offset + 2 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v1 = apply_rope_non_interleaved_fp16(
                                    v1,
                                    p1,
                                    sin_reg[2],
                                    cos_reg[2],
                                    sin_reg[3],
                                    cos_reg[3],
                                    rope_sign,
                                )
                            else:
                                out_v1 = apply_rope_non_interleaved_bf16(
                                    v1,
                                    p1,
                                    sin_reg[2],
                                    cos_reg[2],
                                    sin_reg[3],
                                    cos_reg[3],
                                    rope_sign,
                                )
                        if elem_offset + 4 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v2 = apply_rope_non_interleaved_fp16(
                                    v2,
                                    p2,
                                    sin_reg[4],
                                    cos_reg[4],
                                    sin_reg[5],
                                    cos_reg[5],
                                    rope_sign,
                                )
                            else:
                                out_v2 = apply_rope_non_interleaved_bf16(
                                    v2,
                                    p2,
                                    sin_reg[4],
                                    cos_reg[4],
                                    sin_reg[5],
                                    cos_reg[5],
                                    rope_sign,
                                )
                        if elem_offset + 6 < rotary_dim:
                            if cutlass.const_expr(is_fp16):
                                out_v3 = apply_rope_non_interleaved_fp16(
                                    v3,
                                    p3,
                                    sin_reg[6],
                                    cos_reg[6],
                                    sin_reg[7],
                                    cos_reg[7],
                                    rope_sign,
                                )
                            else:
                                out_v3 = apply_rope_non_interleaved_bf16(
                                    v3,
                                    p3,
                                    sin_reg[6],
                                    cos_reg[6],
                                    sin_reg[7],
                                    cos_reg[7],
                                    rope_sign,
                                )

                    st_global_v4_u32(k_rope_ptr, out_v0, out_v1, out_v2, out_v3)


class RopeKernelCosSinCache:
    """
    RoPE kernel that uses precomputed cos/sin cache.

    This kernel loads cos/sin values from a precomputed cache instead of
    computing them on the fly. This is compatible with vLLM/SGLang style APIs.

    Cache layout:
    - Shape: (max_seq_len, rotary_dim)
    - cos_sin_cache[pos, :rotary_dim//2] = cos values
    - cos_sin_cache[pos, rotary_dim//2:] = sin values

    Uses head-parallel approach: grid.y = total_heads, each block processes
    one token + one head combination.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        head_dim: int,
        rotary_dim: int,
        interleave: bool,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.half_rotary = rotary_dim // 2
        self.interleave = interleave

        self.elems_per_thread = 8
        self.bdx = head_dim // self.elems_per_thread
        self.num_threads = max(128, self.bdx)
        self.bdy = self.num_threads // self.bdx

        self.is_fp16 = dtype.width == 16 and dtype == cutlass.Float16

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        q_rope: cute.Tensor,
        k_rope: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        pos_ids: cute.Tensor,
        nnz: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
        stream,
    ):
        tokens_per_block = self.bdy
        num_token_blocks = (nnz + tokens_per_block - 1) // tokens_per_block
        total_heads = num_qo_heads + num_kv_heads

        # Head-parallel: each block processes one token + one head
        self.kernel(
            q,
            k,
            q_rope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            nnz,
            num_qo_heads,
            num_kv_heads,
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
        cos_sin_cache: cute.Tensor,
        pos_ids: cute.Tensor,
        nnz: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        head_dim = self.head_dim
        rotary_dim = self.rotary_dim
        half_rotary = self.half_rotary
        elems_per_thread = self.elems_per_thread
        bdy = self.bdy
        is_fp16 = self.is_fp16
        interleave = self.interleave

        token_idx = bidx * bdy + tidy
        elem_offset = tidx * elems_per_thread

        # Head-parallel: bidy selects the head
        is_q_head = bidy < num_qo_heads
        q_head_idx = bidy
        k_head_idx = bidy - num_qo_heads

        # For non-interleaved: pre-compute pair offset and sign
        pair_elem_offset = Int32(0)
        rope_sign = Float32(0.0)
        if not interleave:
            in_first_half = elem_offset < half_rotary
            if in_first_half:
                pair_elem_offset = elem_offset + half_rotary
                rope_sign = Float32(-1.0)
            if not in_first_half:
                pair_elem_offset = elem_offset - half_rotary
                rope_sign = Float32(1.0)

        if token_idx < nnz:
            pos = pos_ids[token_idx]

            # Load cos/sin from cache using vectorized PTX loads
            cos_reg = cute.make_rmem_tensor((8,), Float32)
            sin_reg = cute.make_rmem_tensor((8,), Float32)

            # Initialize with defaults (for elements outside rotary_dim)
            for i in cutlass.range_constexpr(8):
                cos_reg[i] = Float32(1.0)
                sin_reg[i] = Float32(0.0)

            # Cache layout: (max_seq_len, rotary_dim), row-major
            # cos at [pos, 0:half_rotary], sin at [pos, half_rotary:rotary_dim]
            cache_row_base = pos * rotary_dim

            if interleave:
                # Interleaved: load cos/sin for pairs
                cache_base = elem_offset // 2

                if elem_offset < rotary_dim:
                    cos_ptr = get_ptr_as_int64(
                        cos_sin_cache, cache_row_base + cache_base
                    )
                    sin_ptr = get_ptr_as_int64(
                        cos_sin_cache, cache_row_base + half_rotary + cache_base
                    )
                    cos_reg[0], cos_reg[1], cos_reg[2], cos_reg[3] = ld_global_v4_f32(
                        cos_ptr
                    )
                    sin_reg[0], sin_reg[1], sin_reg[2], sin_reg[3] = ld_global_v4_f32(
                        sin_ptr
                    )
            else:
                # Non-interleaved: load vec_size consecutive cos/sin values
                cache_base = elem_offset % half_rotary

                if elem_offset < rotary_dim:
                    cos_ptr_0 = get_ptr_as_int64(
                        cos_sin_cache, cache_row_base + cache_base
                    )
                    sin_ptr_0 = get_ptr_as_int64(
                        cos_sin_cache, cache_row_base + half_rotary + cache_base
                    )
                    cos_reg[0], cos_reg[1], cos_reg[2], cos_reg[3] = ld_global_v4_f32(
                        cos_ptr_0
                    )
                    sin_reg[0], sin_reg[1], sin_reg[2], sin_reg[3] = ld_global_v4_f32(
                        sin_ptr_0
                    )

                    if elem_offset + 4 < rotary_dim:
                        cos_ptr_1 = get_ptr_as_int64(
                            cos_sin_cache, cache_row_base + cache_base + 4
                        )
                        sin_ptr_1 = get_ptr_as_int64(
                            cos_sin_cache, cache_row_base + half_rotary + cache_base + 4
                        )
                        cos_reg[4], cos_reg[5], cos_reg[6], cos_reg[7] = (
                            ld_global_v4_f32(cos_ptr_1)
                        )
                        sin_reg[4], sin_reg[5], sin_reg[6], sin_reg[7] = (
                            ld_global_v4_f32(sin_ptr_1)
                        )

            # Process Q head (if this block handles a Q head)
            if is_q_head:
                base_offset = (
                    token_idx * num_qo_heads + q_head_idx
                ) * head_dim + elem_offset
                q_ptr = get_ptr_as_int64(q, base_offset)
                q_rope_ptr = get_ptr_as_int64(q_rope, base_offset)

                v0, v1, v2, v3 = ld_global_v4_u32(q_ptr)
                out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                if interleave:
                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_interleaved_fp16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                        else:
                            out_v0 = apply_rope_interleaved_bf16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_interleaved_fp16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                        else:
                            out_v1 = apply_rope_interleaved_bf16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_interleaved_fp16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                        else:
                            out_v2 = apply_rope_interleaved_bf16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_interleaved_fp16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                        else:
                            out_v3 = apply_rope_interleaved_bf16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                else:
                    # Non-interleaved: need to load paired elements
                    pair_offset = (
                        token_idx * num_qo_heads + q_head_idx
                    ) * head_dim + pair_elem_offset
                    pair_ptr = get_ptr_as_int64(q, pair_offset)
                    p0, p1, p2, p3 = ld_global_v4_u32(pair_ptr)

                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_non_interleaved_fp16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                        else:
                            out_v0 = apply_rope_non_interleaved_bf16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_non_interleaved_fp16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                        else:
                            out_v1 = apply_rope_non_interleaved_bf16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_non_interleaved_fp16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                        else:
                            out_v2 = apply_rope_non_interleaved_bf16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_non_interleaved_fp16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )
                        else:
                            out_v3 = apply_rope_non_interleaved_bf16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )

                st_global_v4_u32(q_rope_ptr, out_v0, out_v1, out_v2, out_v3)

            # Process K head (if this block handles a K head)
            if not is_q_head:
                base_offset = (
                    token_idx * num_kv_heads + k_head_idx
                ) * head_dim + elem_offset
                k_ptr = get_ptr_as_int64(k, base_offset)
                k_rope_ptr = get_ptr_as_int64(k_rope, base_offset)

                v0, v1, v2, v3 = ld_global_v4_u32(k_ptr)
                out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                if interleave:
                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_interleaved_fp16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                        else:
                            out_v0 = apply_rope_interleaved_bf16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_interleaved_fp16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                        else:
                            out_v1 = apply_rope_interleaved_bf16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_interleaved_fp16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                        else:
                            out_v2 = apply_rope_interleaved_bf16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_interleaved_fp16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                        else:
                            out_v3 = apply_rope_interleaved_bf16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                else:
                    # Non-interleaved: need to load paired elements
                    pair_offset = (
                        token_idx * num_kv_heads + k_head_idx
                    ) * head_dim + pair_elem_offset
                    pair_ptr = get_ptr_as_int64(k, pair_offset)
                    p0, p1, p2, p3 = ld_global_v4_u32(pair_ptr)

                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_non_interleaved_fp16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                        else:
                            out_v0 = apply_rope_non_interleaved_bf16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_non_interleaved_fp16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                        else:
                            out_v1 = apply_rope_non_interleaved_bf16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_non_interleaved_fp16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                        else:
                            out_v2 = apply_rope_non_interleaved_bf16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_non_interleaved_fp16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )
                        else:
                            out_v3 = apply_rope_non_interleaved_bf16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )

                st_global_v4_u32(k_rope_ptr, out_v0, out_v1, out_v2, out_v3)


class RopeKernelCosSinCacheSeqHeads:
    """
    Sequential-heads RoPE kernel using precomputed cos/sin cache.

    This version mirrors RopeKernelSeqHeads structure exactly:
    - grid.y = 1 (each block processes all heads for a token)
    - Uses for loops to iterate over heads
    - Loads cos/sin from cache ONCE per token, reuses across heads

    This reduces memory traffic compared to head-parallel approach.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        head_dim: int,
        rotary_dim: int,
        interleave: bool,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.half_rotary = rotary_dim // 2
        self.interleave = interleave

        self.elems_per_thread = 8
        self.bdx = head_dim // self.elems_per_thread
        self.num_threads = max(128, self.bdx)
        self.bdy = self.num_threads // self.bdx

        self.is_fp16 = dtype.width == 16 and dtype == cutlass.Float16

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        q_rope: cute.Tensor,
        k_rope: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        pos_ids: cute.Tensor,
        nnz: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
        stream,
    ):
        tokens_per_block = self.bdy
        num_token_blocks = (nnz + tokens_per_block - 1) // tokens_per_block

        # Sequential heads: grid.y = 1
        self.kernel(
            q,
            k,
            q_rope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            nnz,
            num_qo_heads,
            num_kv_heads,
        ).launch(
            grid=[num_token_blocks, 1, 1],
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
        cos_sin_cache: cute.Tensor,
        pos_ids: cute.Tensor,
        nnz: Int32,
        num_qo_heads: Int32,
        num_kv_heads: Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        head_dim = self.head_dim
        rotary_dim = self.rotary_dim
        half_rotary = self.half_rotary
        elems_per_thread = self.elems_per_thread
        bdy = self.bdy
        is_fp16 = self.is_fp16
        interleave = self.interleave

        token_idx = bidx * bdy + tidy
        elem_offset = tidx * elems_per_thread

        # For non-interleaved: pre-compute pair offset and sign
        pair_elem_offset = Int32(0)
        rope_sign = Float32(0.0)
        if not interleave:
            in_first_half = elem_offset < half_rotary
            if in_first_half:
                pair_elem_offset = elem_offset + half_rotary
                rope_sign = Float32(-1.0)
            if not in_first_half:
                pair_elem_offset = elem_offset - half_rotary
                rope_sign = Float32(1.0)

        if token_idx < nnz:
            pos = pos_ids[token_idx]

            # Load cos/sin from cache ONCE for this token (reused for all heads!)
            # Use vectorized loads (ld_global_v4_f32) matching head-parallel kernel
            sin_reg = cute.make_rmem_tensor((8,), Float32)
            cos_reg = cute.make_rmem_tensor((8,), Float32)

            # Initialize registers with defaults (for elements outside rotary_dim)
            for i in cutlass.range_constexpr(8):
                cos_reg[i] = Float32(1.0)
                sin_reg[i] = Float32(0.0)

            # Cache layout: (max_seq_len, rotary_dim), row-major
            cache_row_base = pos * rotary_dim

            if interleave:
                # Interleaved: load 4 cos/sin values
                cache_base = elem_offset // 2

                if elem_offset < rotary_dim:
                    cos_ptr = get_ptr_as_int64(
                        cos_sin_cache, cache_row_base + cache_base
                    )
                    sin_ptr = get_ptr_as_int64(
                        cos_sin_cache, cache_row_base + half_rotary + cache_base
                    )
                    cos_reg[0], cos_reg[1], cos_reg[2], cos_reg[3] = ld_global_v4_f32(
                        cos_ptr
                    )
                    sin_reg[0], sin_reg[1], sin_reg[2], sin_reg[3] = ld_global_v4_f32(
                        sin_ptr
                    )
            else:
                # Non-interleaved: load 8 cos/sin values
                cache_base = elem_offset % half_rotary

                if elem_offset < rotary_dim:
                    cos_ptr_0 = get_ptr_as_int64(
                        cos_sin_cache, cache_row_base + cache_base
                    )
                    sin_ptr_0 = get_ptr_as_int64(
                        cos_sin_cache, cache_row_base + half_rotary + cache_base
                    )
                    cos_reg[0], cos_reg[1], cos_reg[2], cos_reg[3] = ld_global_v4_f32(
                        cos_ptr_0
                    )
                    sin_reg[0], sin_reg[1], sin_reg[2], sin_reg[3] = ld_global_v4_f32(
                        sin_ptr_0
                    )

                    if elem_offset + 4 < rotary_dim:
                        cos_ptr_1 = get_ptr_as_int64(
                            cos_sin_cache, cache_row_base + cache_base + 4
                        )
                        sin_ptr_1 = get_ptr_as_int64(
                            cos_sin_cache, cache_row_base + half_rotary + cache_base + 4
                        )
                        cos_reg[4], cos_reg[5], cos_reg[6], cos_reg[7] = (
                            ld_global_v4_f32(cos_ptr_1)
                        )
                        sin_reg[4], sin_reg[5], sin_reg[6], sin_reg[7] = (
                            ld_global_v4_f32(sin_ptr_1)
                        )

            # Loop over ALL Q heads (reuses sin/cos!)
            for qo_head_idx in range(num_qo_heads):
                base_offset = (
                    token_idx * num_qo_heads + qo_head_idx
                ) * head_dim + elem_offset
                q_ptr = get_ptr_as_int64(q, base_offset)
                q_rope_ptr = get_ptr_as_int64(q_rope, base_offset)

                v0, v1, v2, v3 = ld_global_v4_u32(q_ptr)
                out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                if interleave:
                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_interleaved_fp16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                        else:
                            out_v0 = apply_rope_interleaved_bf16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_interleaved_fp16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                        else:
                            out_v1 = apply_rope_interleaved_bf16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_interleaved_fp16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                        else:
                            out_v2 = apply_rope_interleaved_bf16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_interleaved_fp16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                        else:
                            out_v3 = apply_rope_interleaved_bf16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                else:
                    # Non-interleaved: load paired elements
                    pair_offset = (
                        token_idx * num_qo_heads + qo_head_idx
                    ) * head_dim + pair_elem_offset
                    pair_ptr = get_ptr_as_int64(q, pair_offset)
                    p0, p1, p2, p3 = ld_global_v4_u32(pair_ptr)

                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_non_interleaved_fp16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                        else:
                            out_v0 = apply_rope_non_interleaved_bf16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_non_interleaved_fp16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                        else:
                            out_v1 = apply_rope_non_interleaved_bf16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_non_interleaved_fp16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                        else:
                            out_v2 = apply_rope_non_interleaved_bf16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_non_interleaved_fp16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )
                        else:
                            out_v3 = apply_rope_non_interleaved_bf16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )

                st_global_v4_u32(q_rope_ptr, out_v0, out_v1, out_v2, out_v3)

            # Loop over ALL K heads (also reuses sin/cos!)
            for kv_head_idx in range(num_kv_heads):
                base_offset = (
                    token_idx * num_kv_heads + kv_head_idx
                ) * head_dim + elem_offset
                k_ptr = get_ptr_as_int64(k, base_offset)
                k_rope_ptr = get_ptr_as_int64(k_rope, base_offset)

                v0, v1, v2, v3 = ld_global_v4_u32(k_ptr)
                out_v0, out_v1, out_v2, out_v3 = v0, v1, v2, v3

                if interleave:
                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_interleaved_fp16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                        else:
                            out_v0 = apply_rope_interleaved_bf16(
                                v0, sin_reg[0], cos_reg[0]
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_interleaved_fp16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                        else:
                            out_v1 = apply_rope_interleaved_bf16(
                                v1, sin_reg[1], cos_reg[1]
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_interleaved_fp16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                        else:
                            out_v2 = apply_rope_interleaved_bf16(
                                v2, sin_reg[2], cos_reg[2]
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_interleaved_fp16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                        else:
                            out_v3 = apply_rope_interleaved_bf16(
                                v3, sin_reg[3], cos_reg[3]
                            )
                else:
                    # Non-interleaved: load paired elements
                    pair_offset = (
                        token_idx * num_kv_heads + kv_head_idx
                    ) * head_dim + pair_elem_offset
                    pair_ptr = get_ptr_as_int64(k, pair_offset)
                    p0, p1, p2, p3 = ld_global_v4_u32(pair_ptr)

                    if elem_offset < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v0 = apply_rope_non_interleaved_fp16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                        else:
                            out_v0 = apply_rope_non_interleaved_bf16(
                                v0,
                                p0,
                                sin_reg[0],
                                cos_reg[0],
                                sin_reg[1],
                                cos_reg[1],
                                rope_sign,
                            )
                    if elem_offset + 2 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v1 = apply_rope_non_interleaved_fp16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                        else:
                            out_v1 = apply_rope_non_interleaved_bf16(
                                v1,
                                p1,
                                sin_reg[2],
                                cos_reg[2],
                                sin_reg[3],
                                cos_reg[3],
                                rope_sign,
                            )
                    if elem_offset + 4 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v2 = apply_rope_non_interleaved_fp16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                        else:
                            out_v2 = apply_rope_non_interleaved_bf16(
                                v2,
                                p2,
                                sin_reg[4],
                                cos_reg[4],
                                sin_reg[5],
                                cos_reg[5],
                                rope_sign,
                            )
                    if elem_offset + 6 < rotary_dim:
                        if cutlass.const_expr(is_fp16):
                            out_v3 = apply_rope_non_interleaved_fp16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )
                        else:
                            out_v3 = apply_rope_non_interleaved_bf16(
                                v3,
                                p3,
                                sin_reg[6],
                                cos_reg[6],
                                sin_reg[7],
                                cos_reg[7],
                                rope_sign,
                            )

                st_global_v4_u32(k_rope_ptr, out_v0, out_v1, out_v2, out_v3)


__all__ = [
    "RopeKernelNonInterleavedVec",
    "RopeKernelInterleavedVec",
    "RopeKernelSeqHeads",
    "RopeKernelWithIndptr",
    "RopeKernelCosSinCache",
    "RopeKernelCosSinCacheSeqHeads",
]
