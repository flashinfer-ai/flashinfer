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
// MiniMax-H3 RMSNorm + indexed AdaLN + NVFP4 activation quantization kernel for sm_100a and sm_103a
// (Blackwell, compute capability 10.0 and 10.3): the one-warp-per-row norm stage shared by the MLP
// block (cake_minimax_h3_mlp_sm10x.cu) and the FC1+SwiGLU operator
// (cake_minimax_h3_fc1_swiglu_sm10x.cu). Both translation units include this header where the
// kernel stood, after their shared device helpers (which the kernel calls); the kernel's helper
// #define block is undefined again after the kernel, as it was in the translation units.
#ifndef CAKE_MINIMAX_H3_NORM_ADALN_NVFP4_SM10X_CUH_
#define CAKE_MINIMAX_H3_NORM_ADALN_NVFP4_SM10X_CUH_

#define THREADS 128
#define HIDDEN 5376
#define ROWS_PER_CTA 4
#define VECS_PER_LANE 21
#define SF_K_TILES 84
#define SF_TILE_BYTES 512

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_minimax_h3_norm_adaln_nvfp4(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, long long* __restrict__ adaln_index, float* __restrict__ a_global_scale, unsigned int* __restrict__ a_q, uint8_t* __restrict__ a_sf, int M, int adaln_rows, long long adaln_row_stride, float eps)
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
        unsigned long long word_base = row_base >> 3;
        int m_tile = row / 128;
        int row_in_tile = row - m_tile * 128;
        unsigned long long sf_row_base = (unsigned long long)m_tile * (unsigned long long)SF_K_TILES * (unsigned long long)SF_TILE_BYTES + (unsigned long long)(row_in_tile & 31) * 16 + (unsigned long long)(row_in_tile >> 5) * 4;
        int pair_lane = lane & 1;
        int block_in_vec = lane >> 1;
        unsigned long long sf_lane_base = sf_row_base + (unsigned long long)(block_in_vec >> 2) * (unsigned long long)SF_TILE_BYTES + (unsigned long long)(block_in_vec & 3);
        float g = a_global_scale[0];
        float _rcp_0 = approx_rcp(g);
        float inv_g = _rcp_0;
        float _rcp_1 = approx_rcp(6.0f);
        float sixth = _rcp_1;
        float zero_f32 = 0.0f;
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
                        float rounded_pair_0[2];
                        rounded_pair_0[0] = 0.0f;
                        rounded_pair_0[1] = _fma_0;
                        unsigned int rounded_word_1[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_pair_0[_lp*2 + 0], rounded_pair_0[_lp*2+1 + 0]));
                            rounded_word_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        float rounded_2 = 0.0f;
                        rounded_2 = reinterpret_cast<float*>(&rounded_word_1[0])[0];
                        vals[2 * j_1 + h] = rounded_2;
                    }
                }
                float mags[8];
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    mags[j_2] = vals[j_2];
                }
                float _fabs_0 = fabsf(mags[0]);
                mags[0] = _fabs_0;
                float _fabs_1 = fabsf(mags[1]);
                mags[1] = _fabs_1;
                float _fabs_2 = fabsf(mags[2]);
                mags[2] = _fabs_2;
                float _fabs_3 = fabsf(mags[3]);
                mags[3] = _fabs_3;
                float _fabs_4 = fabsf(mags[4]);
                mags[4] = _fabs_4;
                float _fabs_5 = fabsf(mags[5]);
                mags[5] = _fabs_5;
                float _fabs_6 = fabsf(mags[6]);
                mags[6] = _fabs_6;
                float _fabs_7 = fabsf(mags[7]);
                mags[7] = _fabs_7;
                float mags_max = mags[0];
                #pragma unroll
                for (int _lr = 1; _lr < 8; _lr++) {
                    mags_max = max_noftz(mags_max, mags[_lr]);
                }
                float absmax = mags_max;
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, absmax, 1);
                float o1 = _shfl_xor_0;
                float _max_0 = max_noftz(absmax, o1);
                absmax = _max_0;
                float sf_value = g * (absmax * sixth);
                uint16_t _e4m3x2_f32_0;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(zero_f32), "f"(sf_value));
                uint16_t sf_pair = _e4m3x2_f32_0;
                unsigned int sf_byte = (unsigned int)sf_pair & 255;
                uint32_t _e4m3x2_to_f16x2_0;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(sf_pair)));
                unsigned int sf_f16x2 = _e4m3x2_to_f16x2_0;
                unsigned int sf_f16_lo = sf_f16x2 & 65535;
                float _cvt_f32_f16_0;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(sf_f16_lo)));
                float sf_f = _cvt_f32_f16_0;
                float _rcp_2 = approx_rcp(sf_f * inv_g);
                float out_scale_nonzero = _rcp_2;
                float out_scale = ((absmax == 0.0f) ? 0.0f : out_scale_nonzero);
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_5 = {out_scale, out_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 4; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(vals)[_ls], _scale2_5);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++) {
                    vals[_ls] = vals[_ls] * out_scale;
                }
                #endif
                unsigned int packed[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(vals[0]), "f"(vals[1]), "f"(vals[2]), "f"(vals[3]), "f"(vals[4]), "f"(vals[5]), "f"(vals[6]), "f"(vals[7]));
                *(reinterpret_cast<unsigned int*>(a_q + (word_base + (unsigned long long)(k_1 >> 3))) + (0)) = packed[0];
                if (pair_lane == 0) {
                    *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + (unsigned long long)(i_1 * (4 * SF_TILE_BYTES)))) + (0)) = (unsigned char)(sf_byte);
                }
            }
        } else {
            unsigned int zero_word = 0;
            #pragma unroll
            for (int i_2 = 0; i_2 < VECS_PER_LANE; i_2++) {
                int k_2 = (lane + i_2 * 32) * 8;
                *(reinterpret_cast<unsigned int*>(a_q + (word_base + (unsigned long long)(k_2 >> 3))) + (0)) = zero_word;
                if (pair_lane == 0) {
                    *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + (unsigned long long)(i_2 * (4 * SF_TILE_BYTES)))) + (0)) = (unsigned char)(zero_word);
                }
            }
        }
    }
}

} // extern "C"

#undef HIDDEN
#undef ROWS_PER_CTA
#undef SF_K_TILES
#undef SF_TILE_BYTES
#undef THREADS
#undef VECS_PER_LANE

#endif  // CAKE_MINIMAX_H3_NORM_ADALN_NVFP4_SM10X_CUH_
// clang-format on
