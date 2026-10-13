/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_nvfp4_mla_decode_c8af9a62d9e2cceb7d2a(__nv_bfloat16* __restrict__ o, float* __restrict__ lse, __nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse, int* __restrict__ row_splits, int total_q, int num_heads, int max_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    float NEG_INF = -CUDART_INF_F;
    int q_idx = blockIdx.x;
    int lane_0 = lane;
    int used = row_splits[q_idx];
    int split_stride = total_q * num_heads * 512;
    int h_idx[2];
    int row[2];
    int lse_base[2];
    int o_base[2];
    float lse_a[2];
    float lse_b[2];
    int h_r = 0;
    int row_r = 0;
    int lse_base_r = 0;
    int o_base_r = 0;
    h_r = blockIdx.y * 16 + warp;
    int _min_0 = ((h_r) < (num_heads - 1) ? (h_r) : (num_heads - 1));
    row_r = q_idx * num_heads + _min_0;
    lse_base_r = row_r * max_splits;
    o_base_r = row_r * 512 + lane_0 * 16;
    h_idx[0] = h_r;
    row[0] = row_r;
    lse_base[0] = lse_base_r;
    o_base[0] = o_base_r;
    lse_a[0] = partial_lse[lse_base_r];
    lse_b[0] = partial_lse[lse_base_r + 1];
    float _vec_load_0[8];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_o + o_base_r + 0);
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
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + 8) + 0);
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
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + split_stride) + 0);
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
    float _vec_load_3[8];
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + split_stride + 8) + 0);
        uint4 _vld_3[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_3[_blk] = _vptr_3[_blk];
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
    }
    h_r = blockIdx.y * 16 + 8 + warp;
    int _min_1 = ((h_r) < (num_heads - 1) ? (h_r) : (num_heads - 1));
    row_r = q_idx * num_heads + _min_1;
    lse_base_r = row_r * max_splits;
    o_base_r = row_r * 512 + lane_0 * 16;
    h_idx[1] = h_r;
    row[1] = row_r;
    lse_base[1] = lse_base_r;
    o_base[1] = o_base_r;
    lse_a[1] = partial_lse[lse_base_r];
    lse_b[1] = partial_lse[lse_base_r + 1];
    float _vec_load_4[8];
    {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(partial_o + o_base_r + 0);
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
    float _vec_load_5[8];
    {
        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + 8) + 0);
        uint4 _vld_5[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_5[_blk] = _vptr_5[_blk];
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
    }
    float _vec_load_6[8];
    {
        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + split_stride) + 0);
        uint4 _vld_6[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_6[_blk] = _vptr_6[_blk];
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
            }
        }
    }
    float _vec_load_7[8];
    {
        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + split_stride + 8) + 0);
        uint4 _vld_7[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_7[_blk] = _vptr_7[_blk];
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
            }
        }
    }
    if (used == 2) {
        float acc2[16];
        float la2 = 0.0f;
        float lb2 = 0.0f;
        float lse_max2 = 0.0f;
        float sum_w2 = 0.0f;
        float glse2 = 0.0f;
        float w_a = 0.0f;
        float w_b = 0.0f;
        la2 = lse_a[0];
        lb2 = lse_b[0];
        float _max_0 = max_noftz(la2, lb2);
        lse_max2 = _max_0;
        lse_max2 = ((lse_max2 != NEG_INF) ? lse_max2 : 0.0f);
        float _exp2_0 = approx_exp2(la2 - lse_max2);
        float _exp2_1 = approx_exp2(lb2 - lse_max2);
        sum_w2 = _exp2_0 + _exp2_1;
        float _log2_0;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(sum_w2));
        glse2 = lse_max2 + _log2_0;
        float _exp2_2 = approx_exp2(la2 - glse2);
        w_a = _exp2_2;
        float _exp2_3 = approx_exp2(lb2 - glse2);
        w_b = _exp2_3;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            acc2[j] = 0.0f;
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            acc2[j_1] = acc2[j_1] + _vec_load_0[j_1] * w_a;
            acc2[8 + j_1] = acc2[8 + j_1] + _vec_load_1[j_1] * w_a;
        }
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            acc2[j_2] = acc2[j_2] + _vec_load_2[j_2] * w_b;
            acc2[8 + j_2] = acc2[8 + j_2] + _vec_load_3[j_2] * w_b;
        }
        if (h_idx[0] < num_heads) {
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(lse + row[0]) + (0)) = glse2 * 0.6931471805599453f;
            }
            {
                __nv_bfloat162 _pk[8];
                _pk[0] = __floats2bfloat162_rn(acc2[0 + 0], acc2[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(acc2[0 + 2], acc2[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(acc2[0 + 4], acc2[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(acc2[0 + 6], acc2[0 + 7]);
                _pk[4] = __floats2bfloat162_rn(acc2[0 + 8], acc2[0 + 9]);
                _pk[5] = __floats2bfloat162_rn(acc2[0 + 10], acc2[0 + 11]);
                _pk[6] = __floats2bfloat162_rn(acc2[0 + 12], acc2[0 + 13]);
                _pk[7] = __floats2bfloat162_rn(acc2[0 + 14], acc2[0 + 15]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row[0] * 512 + lane_0 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row[0] * 512 + lane_0 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
            }
        }
        la2 = lse_a[1];
        lb2 = lse_b[1];
        float _max_1 = max_noftz(la2, lb2);
        lse_max2 = _max_1;
        lse_max2 = ((lse_max2 != NEG_INF) ? lse_max2 : 0.0f);
        float _exp2_4 = approx_exp2(la2 - lse_max2);
        float _exp2_5 = approx_exp2(lb2 - lse_max2);
        sum_w2 = _exp2_4 + _exp2_5;
        float _log2_1;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(sum_w2));
        glse2 = lse_max2 + _log2_1;
        float _exp2_6 = approx_exp2(la2 - glse2);
        w_a = _exp2_6;
        float _exp2_7 = approx_exp2(lb2 - glse2);
        w_b = _exp2_7;
        #pragma unroll
        for (int j_3 = 0; j_3 < 16; j_3++) {
            acc2[j_3] = 0.0f;
        }
        #pragma unroll
        for (int j_4 = 0; j_4 < 8; j_4++) {
            acc2[j_4] = acc2[j_4] + _vec_load_4[j_4] * w_a;
            acc2[8 + j_4] = acc2[8 + j_4] + _vec_load_5[j_4] * w_a;
        }
        #pragma unroll
        for (int j_5 = 0; j_5 < 8; j_5++) {
            acc2[j_5] = acc2[j_5] + _vec_load_6[j_5] * w_b;
            acc2[8 + j_5] = acc2[8 + j_5] + _vec_load_7[j_5] * w_b;
        }
        if (h_idx[1] < num_heads) {
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(lse + row[1]) + (0)) = glse2 * 0.6931471805599453f;
            }
            {
                __nv_bfloat162 _pk[8];
                _pk[0] = __floats2bfloat162_rn(acc2[0 + 0], acc2[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(acc2[0 + 2], acc2[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(acc2[0 + 4], acc2[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(acc2[0 + 6], acc2[0 + 7]);
                _pk[4] = __floats2bfloat162_rn(acc2[0 + 8], acc2[0 + 9]);
                _pk[5] = __floats2bfloat162_rn(acc2[0 + 10], acc2[0 + 11]);
                _pk[6] = __floats2bfloat162_rn(acc2[0 + 12], acc2[0 + 13]);
                _pk[7] = __floats2bfloat162_rn(acc2[0 + 14], acc2[0 + 15]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row[1] * 512 + lane_0 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row[1] * 512 + lane_0 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
            }
        }
    } else if (used > 2) {
        float local_lse[2];
        float acc[16];
        float lse_max = NEG_INF;
        float sum_w = 0.0f;
        float global_lse2 = 0.0f;
        if (h_idx[0] < num_heads) {
            lse_base_r = lse_base[0];
            o_base_r = o_base[0];
            lse_max = NEG_INF;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int sk = lane_0 + i * 32;
                float lv = ((sk < used) ? partial_lse[lse_base_r + sk] : NEG_INF);
                local_lse[i] = lv;
                float _max_2 = max_noftz(lse_max, lv);
                lse_max = _max_2;
            }
            float _warp_reduce_0 = lse_max;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
            lse_max = _warp_reduce_0;
            lse_max = ((lse_max != NEG_INF) ? lse_max : 0.0f);
            sum_w = 0.0f;
            #pragma unroll
            for (int i_1 = 0; i_1 < 2; i_1++) {
                float _exp2_8 = approx_exp2(local_lse[i_1] - lse_max);
                sum_w = sum_w + _exp2_8;
            }
            float _warp_reduce_1 = sum_w;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
            sum_w = _warp_reduce_1;
            float _log2_2;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(sum_w));
            global_lse2 = lse_max + _log2_2;
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(lse + row[0]) + (0)) = global_lse2 * 0.6931471805599453f;
            }
            #pragma unroll
            for (int j_6 = 0; j_6 < 16; j_6++) {
                acc[j_6] = 0.0f;
            }
            #pragma unroll 1
            for (int i_2 = 0; i_2 < used; i_2++) {
                float _exp2_9 = approx_exp2(partial_lse[lse_base_r + i_2] - global_lse2);
                float w_i = _exp2_9;
                #pragma unroll
                for (int j8 = 0; j8 < 16; j8 += 8) {
                    float _vec_load_8[8];
                    {
                        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + i_2 * split_stride + j8) + 0);
                        uint4 _vld_8[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_8[_blk] = _vptr_8[_blk];
                            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 8; j_7++) {
                        acc[j8 + j_7] = acc[j8 + j_7] + _vec_load_8[j_7] * w_i;
                    }
                }
            }
            {
                __nv_bfloat162 _pk[8];
                _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(acc[0 + 4], acc[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(acc[0 + 6], acc[0 + 7]);
                _pk[4] = __floats2bfloat162_rn(acc[0 + 8], acc[0 + 9]);
                _pk[5] = __floats2bfloat162_rn(acc[0 + 10], acc[0 + 11]);
                _pk[6] = __floats2bfloat162_rn(acc[0 + 12], acc[0 + 13]);
                _pk[7] = __floats2bfloat162_rn(acc[0 + 14], acc[0 + 15]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row[0] * 512 + lane_0 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row[0] * 512 + lane_0 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
            }
        }
        if (h_idx[1] < num_heads) {
            lse_base_r = lse_base[1];
            o_base_r = o_base[1];
            lse_max = NEG_INF;
            #pragma unroll
            for (int i_3 = 0; i_3 < 2; i_3++) {
                int sk_1 = lane_0 + i_3 * 32;
                float lv_1 = ((sk_1 < used) ? partial_lse[lse_base_r + sk_1] : NEG_INF);
                local_lse[i_3] = lv_1;
                float _max_3 = max_noftz(lse_max, lv_1);
                lse_max = _max_3;
            }
            float _warp_reduce_2 = lse_max;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_2 = max_noftz(_warp_reduce_2, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset));
            lse_max = _warp_reduce_2;
            lse_max = ((lse_max != NEG_INF) ? lse_max : 0.0f);
            sum_w = 0.0f;
            #pragma unroll
            for (int i_4 = 0; i_4 < 2; i_4++) {
                float _exp2_10 = approx_exp2(local_lse[i_4] - lse_max);
                sum_w = sum_w + _exp2_10;
            }
            float _warp_reduce_3 = sum_w;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
            sum_w = _warp_reduce_3;
            float _log2_3;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(sum_w));
            global_lse2 = lse_max + _log2_3;
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(lse + row[1]) + (0)) = global_lse2 * 0.6931471805599453f;
            }
            #pragma unroll
            for (int j_8 = 0; j_8 < 16; j_8++) {
                acc[j_8] = 0.0f;
            }
            #pragma unroll 1
            for (int i_5 = 0; i_5 < used; i_5++) {
                float _exp2_11 = approx_exp2(partial_lse[lse_base_r + i_5] - global_lse2);
                float w_i_1 = _exp2_11;
                #pragma unroll
                for (int j8_1 = 0; j8_1 < 16; j8_1 += 8) {
                    float _vec_load_9[8];
                    {
                        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + i_5 * split_stride + j8_1) + 0);
                        uint4 _vld_9[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_9[_blk] = _vptr_9[_blk];
                            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 8; j_9++) {
                        acc[j8_1 + j_9] = acc[j8_1 + j_9] + _vec_load_9[j_9] * w_i_1;
                    }
                }
            }
            {
                __nv_bfloat162 _pk[8];
                _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(acc[0 + 4], acc[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(acc[0 + 6], acc[0 + 7]);
                _pk[4] = __floats2bfloat162_rn(acc[0 + 8], acc[0 + 9]);
                _pk[5] = __floats2bfloat162_rn(acc[0 + 10], acc[0 + 11]);
                _pk[6] = __floats2bfloat162_rn(acc[0 + 12], acc[0 + 13]);
                _pk[7] = __floats2bfloat162_rn(acc[0 + 14], acc[0 + 15]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row[1] * 512 + lane_0 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row[1] * 512 + lane_0 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
            }
        }
    }
}

} // extern "C"
