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
#define SMEM_TOTAL 128
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_quant_pre_attention_e6833a4a6fa95d3f470f(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, int* __restrict__ adaln_index, unsigned int* __restrict__ act_q, uint8_t* __restrict__ act_sf, float* __restrict__ act_global_scale, int M, float eps)
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

    // === Task calls (declaration order) ===
    float global_scale = act_global_scale[0];
    #pragma unroll 1
    for (int row = bid; row < M; row += num_bids) {
        float x_vals[64];
        float total = 0.0f;
        for (int i = 0; i < 2; i++) {
            int slot = tid + i * 128;
            for (int e = 0; e < 32; e++) {
                x_vals[i * 32 + e] = 0.0f;
            }
            if (slot < 168) {
                int col = slot * 32;
                for (int q = 0; q < 4; q++) {
                    float _vec_load_0[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (row * 5376 + col + q * 8) + 0);
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
                    for (int e_1 = 0; e_1 < 8; e_1++) {
                        float xv = _vec_load_0[e_1];
                        x_vals[i * 32 + q * 8 + e_1] = xv;
                        total = __fmaf_rn(xv, xv, total);
                    }
                }
            }
        }
        for (int stage = 0; stage < 5; stage++) {
            float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, total, 16 >> stage);
            total += _shfl_xor_sync_1;
        }
        if (lane == 0) {
            partials[warp] = total;
        }
        __syncthreads();
        total = ((lane < 4) ? partials[lane] : 0.0f);
        for (int stage_1 = 0; stage_1 < 2; stage_1++) {
            float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, total, 2 >> stage_1);
            total += _shfl_xor_sync_2;
        }
        float _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, total, 0);
        total = _shfl_sync_3;
        float _fdiv_4 = __fdiv_rn(total, 5376.0f);
        float rstd = rsqrtf(_fdiv_4 + eps);
        int idx = adaln_index[row];
        int valid = ((idx >= 0 && idx < 9) ? 1 : 0);
        int idx_c = ((valid == 1) ? idx : 0);
        for (int i_1 = 0; i_1 < 2; i_1++) {
            int slot_1 = tid + i_1 * 128;
            if (slot_1 < 168) {
                int col_1 = slot_1 * 32;
                unsigned int words[4];
                for (int blk = 0; blk < 2; blk++) {
                    float a_blk[16];
                    float amax = 0.0f;
                    for (int h = 0; h < 2; h++) {
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x_norm_weight + (col_1 + (blk * 16 + h * 8)) + 0);
                            uint4 _vld_5[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_5[_blk] = _vptr_5[_blk];
                                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_2[8];
                        {
                            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(adaln_scale + (idx_c * 5376 + col_1 + (blk * 16 + h * 8)) + 0);
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
                            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(adaln_shift + (idx_c * 5376 + col_1 + (blk * 16 + h * 8)) + 0);
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
                        float n_raw[8];
                        float sp1_raw[8];
                        for (int e_2 = 0; e_2 < 8; e_2++) {
                            float wv = _vec_load_1[e_2];
                            float sv = _vec_load_2[e_2];
                            n_raw[e_2] = x_vals[i_1 * 32 + (blk * 16 + h * 8) + e_2] * rstd * wv;
                            sp1_raw[e_2] = sv + 1.0f;
                        }
                        unsigned int n_raw_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(n_raw[_lp*2 + 0], n_raw[_lp*2+1 + 0]));
                            n_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _unpack_bf16_0[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(n_raw_bf16[_pair]) << 16);
                            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(n_raw_bf16[_pair]) & 0xffff0000u);
                        }
                        unsigned int sp1_raw_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sp1_raw[_lp*2 + 0], sp1_raw[_lp*2+1 + 0]));
                            sp1_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _unpack_bf16_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sp1_raw_bf16[_pair]) << 16);
                            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sp1_raw_bf16[_pair]) & 0xffff0000u);
                        }
                        float a_raw[8];
                        for (int e_3 = 0; e_3 < 8; e_3++) {
                            float bv = _vec_load_3[e_3];
                            a_raw[e_3] = __fmaf_rn(_unpack_bf16_0[e_3], _unpack_bf16_1[e_3], bv);
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
                        for (int e_4 = 0; e_4 < 8; e_4++) {
                            float av = ((valid == 1) ? _unpack_bf16_2[e_4] : 0.0f);
                            a_blk[h * 8 + e_4] = av;
                            amax = fmaxf(amax, fabsf(av));
                        }
                    }
                    float sf_val = amax * 0.16666667f * global_scale;
                    int block_idx = slot_1 * 2 + blk;
                    uint16_t _cvt_satfinite_e4m3x2_f32_8;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_8) : "f"(0.0f), "f"(sf_val));
                    *reinterpret_cast<unsigned char*>(act_sf + (row * 336 + block_idx)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_8 & 127));
                    uint16_t _cvt_satfinite_e4m3x2_f32_9;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_9) : "f"(0.0f), "f"(sf_val));
                    uint16_t _e4m3x2_decode_10 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_9 & 0xFFu);
                    uint32_t _f16x2_decode_10;
                    float sf_dec;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_10) : "h"(_e4m3x2_decode_10));
                    uint16_t _f16_decode_10 = (uint16_t)_f16x2_decode_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_dec) : "h"(_f16_decode_10));
                    float _fdiv_11 = __fdiv_rn(global_scale, sf_dec);
                    float out_scale = ((sf_dec != 0.0f) ? _fdiv_11 : 0.0f);
                    for (int e_5 = 0; e_5 < 16; e_5++) {
                        a_blk[e_5] = a_blk[e_5] * out_scale;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(words[(blk * 2) + 0]) : "f"(a_blk[0]), "f"(a_blk[1]), "f"(a_blk[2]), "f"(a_blk[3]), "f"(a_blk[4]), "f"(a_blk[5]), "f"(a_blk[6]), "f"(a_blk[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(words[(blk * 2) + 1]) : "f"(a_blk[8]), "f"(a_blk[9]), "f"(a_blk[10]), "f"(a_blk[11]), "f"(a_blk[12]), "f"(a_blk[13]), "f"(a_blk[14]), "f"(a_blk[15]));
                }
                reinterpret_cast<int4*>(act_q + ((row * 2688 + slot_1 * 16) / 4))[0] = reinterpret_cast<int4*>(words)[0];
            }
        }
        __syncthreads();
    }
}

} // extern "C"
