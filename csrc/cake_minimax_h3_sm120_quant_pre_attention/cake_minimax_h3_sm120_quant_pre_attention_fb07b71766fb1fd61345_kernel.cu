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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_1_STAGES 1
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 32
#define SMEM_PARTIALS_STRIDE 32
#define SMEM_AMAX_PARTIALS_OFF 32
#define SMEM_AMAX_PARTIALS_STAGE_BYTES 32
#define SMEM_AMAX_PARTIALS_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_quant_pre_attention_fb07b71766fb1fd61345(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, int* __restrict__ adaln_index, unsigned int* __restrict__ act_q, float* __restrict__ act_scale, int M, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;
    float* amax_partials = reinterpret_cast<float*>(smem_raw + 32);
    const int amax_partials_addr = smem + 32;

    // === Task calls (declaration order) ===
    #pragma unroll 1
    for (int row = bid; row < M; row += num_bids) {
        float x_vals[48];
        float a_vals[48];
        float total = 0.0f;
        for (int i = 0; i < 3; i++) {
            int chunk = tid + i * 128;
            for (int e = 0; e < 16; e++) {
                x_vals[i * 16 + e] = 0.0f;
            }
            if (chunk < 336) {
                int col = chunk * 16;
                float _vec_load_0[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (row * 5376 + col) + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                            (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float _vec_load_1[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + (row * 5376 + col + 8) + 0);
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
                for (int e_1 = 0; e_1 < 8; e_1++) {
                    float xv0 = _vec_load_0[e_1];
                    float xv1 = _vec_load_1[e_1];
                    x_vals[i * 16 + e_1] = xv0;
                    x_vals[i * 16 + 8 + e_1] = xv1;
                    total += __fmaf_rn(xv0, xv0, xv1 * xv1);
                }
            }
        }
        for (int stage = 0; stage < 5; stage++) {
            float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, total, 16 >> stage);
            total += _shfl_xor_sync_2;
        }
        if (lane == 0) {
            partials[warp] = total;
        }
        __syncthreads();
        total = ((lane < 4) ? partials[lane] : 0.0f);
        for (int stage_1 = 0; stage_1 < 2; stage_1++) {
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, total, 2 >> stage_1);
            total += _shfl_xor_sync_3;
        }
        float _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, total, 0);
        total = _shfl_sync_4;
        float _fdiv_5 = __fdiv_rn(total, 5376.0f);
        float rstd = rsqrtf(_fdiv_5 + eps);
        int idx = adaln_index[row];
        int valid = ((idx >= 0 && idx < 9) ? 1 : 0);
        int idx_c = ((valid == 1) ? idx : 0);
        float amax = 0.0f;
        for (int i_1 = 0; i_1 < 3; i_1++) {
            int chunk_1 = tid + i_1 * 128;
            for (int e_2 = 0; e_2 < 16; e_2++) {
                a_vals[i_1 * 16 + e_2] = 0.0f;
            }
            if (chunk_1 < 336) {
                int col_1 = chunk_1 * 16;
                for (int h = 0; h < 2; h++) {
                    float _vec_load_2[8];
                    {
                        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x_norm_weight + (col_1 + h * 8) + 0);
                        uint4 _vld_6[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_6[_blk] = _vptr_6[_blk];
                            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float _vec_load_3[8];
                    {
                        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(adaln_scale + (idx_c * 5376 + col_1 + h * 8) + 0);
                        uint4 _vld_7[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_7[_blk] = _vptr_7[_blk];
                            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float _vec_load_4[8];
                    {
                        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(adaln_shift + (idx_c * 5376 + col_1 + h * 8) + 0);
                        uint4 _vld_8[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_8[_blk] = _vptr_8[_blk];
                            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float n_raw[8];
                    float sp1_raw[8];
                    for (int e_3 = 0; e_3 < 8; e_3++) {
                        float wv = _vec_load_2[e_3];
                        float sv = _vec_load_3[e_3];
                        n_raw[e_3] = x_vals[i_1 * 16 + h * 8 + e_3] * rstd * wv;
                        sp1_raw[e_3] = sv + 1.0f;
                    }
                    unsigned int n_raw_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(n_raw[_lp*2 + 0], n_raw[_lp*2+1 + 0]));
                        n_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int sp1_raw_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sp1_raw[_lp*2 + 0], sp1_raw[_lp*2+1 + 0]));
                        sp1_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_0[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(n_raw_bf16[_pair]) << 16);
                        (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(n_raw_bf16[_pair]) & 0xffff0000u);
                    }
                    float _unpack_bf16_1[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sp1_raw_bf16[_pair]) << 16);
                        (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sp1_raw_bf16[_pair]) & 0xffff0000u);
                    }
                    float a_raw[8];
                    for (int e_4 = 0; e_4 < 8; e_4++) {
                        float bv = _vec_load_4[e_4];
                        a_raw[e_4] = __fmaf_rn(_unpack_bf16_0[e_4], _unpack_bf16_1[e_4], bv);
                    }
                    unsigned int a_raw_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(a_raw[_lp*2 + 0], a_raw[_lp*2+1 + 0]));
                        a_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_2[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(a_raw_bf16[_pair]) << 16);
                        (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(a_raw_bf16[_pair]) & 0xffff0000u);
                    }
                    for (int e_5 = 0; e_5 < 8; e_5++) {
                        float av = ((valid == 1) ? _unpack_bf16_2[e_5] : 0.0f);
                        a_vals[i_1 * 16 + h * 8 + e_5] = av;
                        amax = fmaxf(amax, fabsf(av));
                    }
                }
            }
        }
        for (int stage_2 = 0; stage_2 < 5; stage_2++) {
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, amax, 16 >> stage_2);
            amax = fmaxf(amax, _shfl_xor_sync_9);
        }
        if (lane == 0) {
            amax_partials[warp] = amax;
        }
        __syncthreads();
        amax = ((lane < 4) ? amax_partials[lane] : 0.0f);
        for (int stage_3 = 0; stage_3 < 2; stage_3++) {
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, amax, 2 >> stage_3);
            amax = fmaxf(amax, _shfl_xor_sync_10);
        }
        float _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, amax, 0);
        amax = _shfl_sync_11;
        amax = fmaxf(amax, 1e-12f);
        float scale = __fdiv_rn(amax, 448.0f);
        if (tid == 0) {
            *(reinterpret_cast<float*>(act_scale + row) + (0)) = scale;
        }
        for (int i_2 = 0; i_2 < 3; i_2++) {
            int chunk_2 = tid + i_2 * 128;
            if (chunk_2 < 336) {
                unsigned int words[4];
                for (int w = 0; w < 4; w++) {
                    float q_a = __fdiv_rn(a_vals[i_2 * 16 + w * 4], scale);
                    float q_b = __fdiv_rn(a_vals[i_2 * 16 + w * 4 + 1], scale);
                    float q_c = __fdiv_rn(a_vals[i_2 * 16 + w * 4 + 2], scale);
                    float q_d = __fdiv_rn(a_vals[i_2 * 16 + w * 4 + 3], scale);
                    uint16_t _cvt_satfinite_e4m3x2_f32_13;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_13) : "f"(q_b), "f"(q_a));
                    uint16_t _cvt_satfinite_e4m3x2_f32_14;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14) : "f"(q_d), "f"(q_c));
                    uint32_t _mov_b32_pack_u16x2_12;
                    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_12) : "h"(_cvt_satfinite_e4m3x2_f32_13), "h"(_cvt_satfinite_e4m3x2_f32_14));
                    words[w] = _mov_b32_pack_u16x2_12;
                }
                reinterpret_cast<int4*>(act_q + ((row * 5376 + chunk_2 * 16) / 4))[0] = reinterpret_cast<int4*>(words)[0];
            }
        }
        __syncthreads();
    }
}

} // extern "C"
