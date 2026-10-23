/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// clang-format off
// MiniMax-H3 RMSNorm + indexed AdaLN (BF16 activation output) kernel for sm_100a and sm_103a
// (Blackwell, compute capability 10.0 and 10.3): the one-warp-per-row norm stage shared by the MLP
// block (cake_minimax_h3_mlp_sm10x.cu) and the FC1+SwiGLU operator
// (cake_minimax_h3_fc1_swiglu_sm10x.cu). Both translation units include this header where the
// kernel stood, after their shared device helpers (which the kernel calls); the kernel's helper
// #define block is undefined again after the kernel, as it was in the translation units.
#ifndef CAKE_MINIMAX_H3_NORM_ADALN_BF16_SM10X_CUH_
#define CAKE_MINIMAX_H3_NORM_ADALN_BF16_SM10X_CUH_

#define THREADS 128
#define HIDDEN 5376
#define ROWS_PER_CTA 4
#define VECS_PER_LANE 21

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_minimax_h3_norm_adaln_bf16(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, long long* __restrict__ adaln_index, __nv_bfloat16* __restrict__ a_out, int M, int adaln_rows, long long adaln_row_stride, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int row = bid * ROWS_PER_CTA + warp;
    int _min_0 = ((row) < (M - 1) ? (row) : (M - 1));
    int load_row = _min_0;
    unsigned long long load_base = (unsigned long long)load_row * (unsigned long long)HIDDEN;
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < VECS_PER_LANE; i++) {
        int k = (lane + i * 32) * 8;
        unsigned int _vec_load_0[4];
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(x + (load_base + (unsigned long long)k) + 0);
            _vec_load_0[0 + 0] = _uv4_0.x;
            _vec_load_0[0 + 1] = _uv4_0.y;
            _vec_load_0[0 + 2] = _uv4_0.z;
            _vec_load_0[0 + 3] = _uv4_0.w;
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float _bf16x2_dot_f32_0;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_bf16x2_dot_f32_0) : "r"(_vec_load_0[j]), "r"(_vec_load_0[j]), "f"(sum_sq));
            sum_sq = _bf16x2_dot_f32_0;
        }
    }
    float _warp_reduce_0 = sum_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    float total = _warp_reduce_0;
    float _rsqrt_0 = rsqrtf(total / (float)HIDDEN + eps);
    float rstd = _rsqrt_0;
    if (row < M) {
        unsigned long long row_base = (unsigned long long)row * (unsigned long long)HIDDEN;
        long long table_row = adaln_index[row];
        if (table_row >= 0 && table_row < (long long)adaln_rows) {
            unsigned long long table_base = (unsigned long long)table_row * (unsigned long long)adaln_row_stride;
            unsigned int bf16_one_pair = 1065369472;
            #pragma unroll
            for (int i_1 = 0; i_1 < VECS_PER_LANE; i_1++) {
                int k_1 = (lane + i_1 * 32) * 8;
                float _vec_load_1[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + (row_base + (unsigned long long)k_1) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float _vec_load_2[8];
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x_norm_weight + k_1 + 0);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
                unsigned int _vec_load_3[4];
                {
                    uint4 _uv4_3 = *reinterpret_cast<const uint4*>(adaln_scale + (table_base + (unsigned long long)k_1) + 0);
                    _vec_load_3[0 + 0] = _uv4_3.x;
                    _vec_load_3[0 + 1] = _uv4_3.y;
                    _vec_load_3[0 + 2] = _uv4_3.z;
                    _vec_load_3[0 + 3] = _uv4_3.w;
                }
                float _vec_load_4[8];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(adaln_shift + (table_base + (unsigned long long)k_1) + 0);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                            (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vals[8];
                #pragma unroll
                for (int j_1 = 0; j_1 < 4; j_1++) {
                    unsigned int s_plus_one_word[1];
                    uint32_t _bf16x2_add_0;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_0) : "r"(_vec_load_3[j_1]), "r"(bf16_one_pair));
                    s_plus_one_word[0] = _bf16x2_add_0;
                    float s_plus_one_word_f32[2];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&s_plus_one_word_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(s_plus_one_word[_pair]) << 16);
                        (&s_plus_one_word_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(s_plus_one_word[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float scaled = rstd * _vec_load_1[2 * j_1 + h];
                        float rounded_pair[2];
                        rounded_pair[0] = 0.0f;
                        rounded_pair[1] = _vec_load_2[2 * j_1 + h] * scaled;
                        unsigned int rounded_word[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_pair[_lp*2 + 0], rounded_pair[_lp*2+1 + 0]));
                            rounded_word[_lp] = *(uint32_t*)&_bf2;
                        }
                        float rounded = 0.0f;
                        rounded = reinterpret_cast<float*>(&rounded_word[0])[0];
                        float norm_value = rounded;
                        float scale_plus_one = s_plus_one_word_f32[h];
                        float _fma_0 = __fmaf_rn(norm_value, scale_plus_one, _vec_load_4[2 * j_1 + h]);
                        vals[2 * j_1 + h] = _fma_0;
                    }
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(a_out + (row_base + (unsigned long long)k_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        } else {
            #pragma unroll
            for (int i_2 = 0; i_2 < VECS_PER_LANE; i_2++) {
                int k_2 = (lane + i_2 * 32) * 8;
                float zeros[8];
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    zeros[j_2] = 0.0f;
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(a_out + (row_base + (unsigned long long)k_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
    }
}

} // extern "C"

#undef HIDDEN
#undef ROWS_PER_CTA
#undef THREADS
#undef VECS_PER_LANE

#endif  // CAKE_MINIMAX_H3_NORM_ADALN_BF16_SM10X_CUH_
// clang-format on
