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
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 4

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
kernel_cake_nvfp4_mla_decode_d3e21f744c0427fef73d(__nv_bfloat16* __restrict__ o, float* __restrict__ lse, __nv_bfloat16* partial_o, float* partial_lse, int* __restrict__ row_splits, int total_q, int num_heads, int max_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    float NEG_INF = -CUDART_INF_F;
    int q_idx = blockIdx.x;
    int lane_0 = lane;
    int used = row_splits[q_idx];
    int split_stride = total_q * num_heads * 512;
    int h_idx[2];
    int row[2];
    int lse_base[2];
    int o_base[2];
    float lse_lane[2];
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
    lse_lane[0] = ((lane_0 < max_splits) ? partial_lse[lse_base_r + lane_0] : NEG_INF);
    float _vec_load_0[8];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(partial_o + o_base_r + 0);
        uint4 _vld_1[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_1[_blk] = _vptr_1[_blk];
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
    }
    float _vec_load_1[8];
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + 8) + 0);
        uint4 _vld_2[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_2[_blk] = _vptr_2[_blk];
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
    }
    float _vec_load_2[8];
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + split_stride) + 0);
        uint4 _vld_3[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_3[_blk] = _vptr_3[_blk];
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
    }
    float _vec_load_3[8];
    {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + split_stride + 8) + 0);
        uint4 _vld_4[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_4[_blk] = _vptr_4[_blk];
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
            }
        }
    }
    h_r = blockIdx.y * 16 + 8 + warp;
    int _min_5 = ((h_r) < (num_heads - 1) ? (h_r) : (num_heads - 1));
    row_r = q_idx * num_heads + _min_5;
    lse_base_r = row_r * max_splits;
    o_base_r = row_r * 512 + lane_0 * 16;
    h_idx[1] = h_r;
    row[1] = row_r;
    lse_base[1] = lse_base_r;
    o_base[1] = o_base_r;
    lse_lane[1] = ((lane_0 < max_splits) ? partial_lse[lse_base_r + lane_0] : NEG_INF);
    float _vec_load_4[8];
    {
        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(partial_o + o_base_r + 0);
        uint4 _vld_6[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_6[_blk] = _vptr_6[_blk];
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
            }
        }
    }
    float _vec_load_5[8];
    {
        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + 8) + 0);
        uint4 _vld_7[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_7[_blk] = _vptr_7[_blk];
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
            }
        }
    }
    float _vec_load_6[8];
    {
        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + split_stride) + 0);
        uint4 _vld_8[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_8[_blk] = _vptr_8[_blk];
            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
            }
        }
    }
    float _vec_load_7[8];
    {
        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + split_stride + 8) + 0);
        uint4 _vld_9[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_9[_blk] = _vptr_9[_blk];
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
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
        float _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, lse_lane[0], 0);
        la2 = _shfl_sync_10;
        float _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, lse_lane[0], 1);
        lb2 = _shfl_sync_11;
        lse_max2 = fmaxf(la2, lb2);
        lse_max2 = ((lse_max2 != NEG_INF) ? lse_max2 : 0.0f);
        float _exp2_12 = approx_exp2(la2 - lse_max2);
        float _exp2_13 = approx_exp2(lb2 - lse_max2);
        sum_w2 = _exp2_12 + _exp2_13;
        float _log2_14;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_14) : "f"(sum_w2));
        glse2 = lse_max2 + _log2_14;
        float _exp2_15 = approx_exp2(la2 - glse2);
        w_a = _exp2_15;
        float _exp2_16 = approx_exp2(lb2 - glse2);
        w_b = _exp2_16;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            acc2[j] = 0.0f;
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            acc2[j_1] = __fmaf_rn(_vec_load_0[j_1], w_a, acc2[j_1]);
            acc2[8 + j_1] = __fmaf_rn(_vec_load_1[j_1], w_a, acc2[8 + j_1]);
        }
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            acc2[j_2] = __fmaf_rn(_vec_load_2[j_2], w_b, acc2[j_2]);
            acc2[8 + j_2] = __fmaf_rn(_vec_load_3[j_2], w_b, acc2[8 + j_2]);
        }
        if (h_idx[0] < num_heads) {
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(lse + row[0]) + (0)) = glse2 * 0.6931472f;
            }
            unsigned int _packed_0[8];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 0], acc2[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 2], acc2[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 4], acc2[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 6], acc2[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 8], acc2[_lp*2+1 + 8]));
                _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 10], acc2[_lp*2+1 + 10]));
                _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 12], acc2[_lp*2+1 + 12]));
                _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 14], acc2[_lp*2+1 + 14]));
                _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(o + (row[0] * 512 + lane_0 * 16))[0] = reinterpret_cast<int4*>(_packed_0)[0];
            reinterpret_cast<int4*>(o + (row[0] * 512 + lane_0 * 16 + 8))[0] = reinterpret_cast<int4*>(_packed_0 + (4))[0];
        }
        float _shfl_sync_17 = __shfl_sync(0xFFFFFFFF, lse_lane[1], 0);
        la2 = _shfl_sync_17;
        float _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, lse_lane[1], 1);
        lb2 = _shfl_sync_18;
        lse_max2 = fmaxf(la2, lb2);
        lse_max2 = ((lse_max2 != NEG_INF) ? lse_max2 : 0.0f);
        float _exp2_19 = approx_exp2(la2 - lse_max2);
        float _exp2_20 = approx_exp2(lb2 - lse_max2);
        sum_w2 = _exp2_19 + _exp2_20;
        float _log2_21;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_21) : "f"(sum_w2));
        glse2 = lse_max2 + _log2_21;
        float _exp2_22 = approx_exp2(la2 - glse2);
        w_a = _exp2_22;
        float _exp2_23 = approx_exp2(lb2 - glse2);
        w_b = _exp2_23;
        #pragma unroll
        for (int j_3 = 0; j_3 < 16; j_3++) {
            acc2[j_3] = 0.0f;
        }
        #pragma unroll
        for (int j_4 = 0; j_4 < 8; j_4++) {
            acc2[j_4] = __fmaf_rn(_vec_load_4[j_4], w_a, acc2[j_4]);
            acc2[8 + j_4] = __fmaf_rn(_vec_load_5[j_4], w_a, acc2[8 + j_4]);
        }
        #pragma unroll
        for (int j_5 = 0; j_5 < 8; j_5++) {
            acc2[j_5] = __fmaf_rn(_vec_load_6[j_5], w_b, acc2[j_5]);
            acc2[8 + j_5] = __fmaf_rn(_vec_load_7[j_5], w_b, acc2[8 + j_5]);
        }
        if (h_idx[1] < num_heads) {
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(lse + row[1]) + (0)) = glse2 * 0.6931472f;
            }
            unsigned int _packed_1[8];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 0], acc2[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 2], acc2[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 4], acc2[_lp*2+1 + 4]));
                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 6], acc2[_lp*2+1 + 6]));
                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 8], acc2[_lp*2+1 + 8]));
                _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 10], acc2[_lp*2+1 + 10]));
                _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 12], acc2[_lp*2+1 + 12]));
                _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 14], acc2[_lp*2+1 + 14]));
                _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(o + (row[1] * 512 + lane_0 * 16))[0] = reinterpret_cast<int4*>(_packed_1)[0];
            reinterpret_cast<int4*>(o + (row[1] * 512 + lane_0 * 16 + 8))[0] = reinterpret_cast<int4*>(_packed_1 + (4))[0];
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
            local_lse[0] = ((lane_0 < used) ? lse_lane[0] : NEG_INF);
            local_lse[1] = NEG_INF;
            if (used > 32) {
                int sk1 = lane_0 + 32;
                local_lse[1] = ((sk1 < used) ? partial_lse[lse_base_r + sk1] : NEG_INF);
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                lse_max = fmaxf(lse_max, local_lse[i]);
            }
            float _warp_reduce_24 = lse_max;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_24 = max_noftz(_warp_reduce_24, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_24, offset));
            lse_max = _warp_reduce_24;
            lse_max = ((lse_max != NEG_INF) ? lse_max : 0.0f);
            sum_w = 0.0f;
            #pragma unroll
            for (int i_1 = 0; i_1 < 2; i_1++) {
                float _exp2_25 = approx_exp2(local_lse[i_1] - lse_max);
                sum_w = sum_w + _exp2_25;
            }
            float _warp_reduce_26 = sum_w;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_26 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_26, offset);
            sum_w = _warp_reduce_26;
            float _log2_27;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_27) : "f"(sum_w));
            global_lse2 = lse_max + _log2_27;
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(lse + row[0]) + (0)) = global_lse2 * 0.6931472f;
            }
            #pragma unroll
            for (int j_6 = 0; j_6 < 16; j_6++) {
                acc[j_6] = 0.0f;
            }
            float _shfl_sync_28 = __shfl_sync(0xFFFFFFFF, lse_lane[0], 0);
            float w_0 = approx_exp2(_shfl_sync_28 - global_lse2);
            float _shfl_sync_29 = __shfl_sync(0xFFFFFFFF, lse_lane[0], 1);
            float w_1 = approx_exp2(_shfl_sync_29 - global_lse2);
            #pragma unroll
            for (int j_7 = 0; j_7 < 8; j_7++) {
                acc[j_7] = __fmaf_rn(_vec_load_0[j_7], w_0, acc[j_7]);
                acc[8 + j_7] = __fmaf_rn(_vec_load_1[j_7], w_0, acc[8 + j_7]);
            }
            #pragma unroll
            for (int j_8 = 0; j_8 < 8; j_8++) {
                acc[j_8] = __fmaf_rn(_vec_load_2[j_8], w_1, acc[j_8]);
                acc[8 + j_8] = __fmaf_rn(_vec_load_3[j_8], w_1, acc[8 + j_8]);
            }
            #pragma unroll 1
            for (int i_2 = 2; i_2 < used - 1; i_2 += 2) {
                float w_i = approx_exp2(partial_lse[lse_base_r + i_2] - global_lse2);
                float w_n = approx_exp2(partial_lse[lse_base_r + i_2 + 1] - global_lse2);
                float _vec_load_8[8];
                {
                    const uint4* _vptr_30 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + i_2 * split_stride) + 0);
                    uint4 _vld_30[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_30[_blk] = _vptr_30[_blk];
                        uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
                            (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float _vec_load_9[8];
                {
                    const uint4* _vptr_31 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + i_2 * split_stride + 8) + 0);
                    uint4 _vld_31[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_31[_blk] = _vptr_31[_blk];
                        uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
                            (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float _vec_load_10[8];
                {
                    const uint4* _vptr_32 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + (i_2 + 1) * split_stride) + 0);
                    uint4 _vld_32[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_32[_blk] = _vptr_32[_blk];
                        uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                            (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float _vec_load_11[8];
                {
                    const uint4* _vptr_33 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + (i_2 + 1) * split_stride + 8) + 0);
                    uint4 _vld_33[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_33[_blk] = _vptr_33[_blk];
                        uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                            (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_9 = 0; j_9 < 8; j_9++) {
                    acc[j_9] = __fmaf_rn(_vec_load_8[j_9], w_i, acc[j_9]);
                    acc[8 + j_9] = __fmaf_rn(_vec_load_9[j_9], w_i, acc[8 + j_9]);
                }
                #pragma unroll
                for (int j_10 = 0; j_10 < 8; j_10++) {
                    acc[j_10] = __fmaf_rn(_vec_load_10[j_10], w_n, acc[j_10]);
                    acc[8 + j_10] = __fmaf_rn(_vec_load_11[j_10], w_n, acc[8 + j_10]);
                }
            }
            if ((used & 1) != 0) {
                int i_last = used - 1;
                float w_l = approx_exp2(partial_lse[lse_base_r + i_last] - global_lse2);
                #pragma unroll
                for (int j8 = 0; j8 < 16; j8 += 8) {
                    float _vec_load_12[8];
                    {
                        const uint4* _vptr_34 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + i_last * split_stride + j8) + 0);
                        uint4 _vld_34[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_34[_blk] = _vptr_34[_blk];
                            uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
                                (&_vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_11 = 0; j_11 < 8; j_11++) {
                        acc[j8 + j_11] = __fmaf_rn(_vec_load_12[j_11], w_l, acc[j8 + j_11]);
                    }
                }
            }
            unsigned int _packed_2[8];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 4], acc[_lp*2+1 + 4]));
                _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 6], acc[_lp*2+1 + 6]));
                _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
                _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 10], acc[_lp*2+1 + 10]));
                _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 12], acc[_lp*2+1 + 12]));
                _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 14], acc[_lp*2+1 + 14]));
                _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(o + (row[0] * 512 + lane_0 * 16))[0] = reinterpret_cast<int4*>(_packed_2)[0];
            reinterpret_cast<int4*>(o + (row[0] * 512 + lane_0 * 16 + 8))[0] = reinterpret_cast<int4*>(_packed_2 + (4))[0];
        }
        if (h_idx[1] < num_heads) {
            lse_base_r = lse_base[1];
            o_base_r = o_base[1];
            lse_max = NEG_INF;
            local_lse[0] = ((lane_0 < used) ? lse_lane[1] : NEG_INF);
            local_lse[1] = NEG_INF;
            if (used > 32) {
                int sk1_1 = lane_0 + 32;
                local_lse[1] = ((sk1_1 < used) ? partial_lse[lse_base_r + sk1_1] : NEG_INF);
            }
            #pragma unroll
            for (int i_3 = 0; i_3 < 2; i_3++) {
                lse_max = fmaxf(lse_max, local_lse[i_3]);
            }
            float _warp_reduce_35 = lse_max;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_35 = max_noftz(_warp_reduce_35, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_35, offset));
            lse_max = _warp_reduce_35;
            lse_max = ((lse_max != NEG_INF) ? lse_max : 0.0f);
            sum_w = 0.0f;
            #pragma unroll
            for (int i_4 = 0; i_4 < 2; i_4++) {
                float _exp2_36 = approx_exp2(local_lse[i_4] - lse_max);
                sum_w = sum_w + _exp2_36;
            }
            float _warp_reduce_37 = sum_w;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_37 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_37, offset);
            sum_w = _warp_reduce_37;
            float _log2_38;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_38) : "f"(sum_w));
            global_lse2 = lse_max + _log2_38;
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(lse + row[1]) + (0)) = global_lse2 * 0.6931472f;
            }
            #pragma unroll
            for (int j_12 = 0; j_12 < 16; j_12++) {
                acc[j_12] = 0.0f;
            }
            #pragma unroll 1
            for (int i_5 = 0; i_5 < used - 1; i_5 += 2) {
                float w_i_1 = approx_exp2(partial_lse[lse_base_r + i_5] - global_lse2);
                float w_n_1 = approx_exp2(partial_lse[lse_base_r + i_5 + 1] - global_lse2);
                float _vec_load_13[8];
                {
                    const uint4* _vptr_39 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + i_5 * split_stride) + 0);
                    uint4 _vld_39[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_39[_blk] = _vptr_39[_blk];
                        uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) << 16);
                            (&_vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float _vec_load_14[8];
                {
                    const uint4* _vptr_40 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + i_5 * split_stride + 8) + 0);
                    uint4 _vld_40[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_40[_blk] = _vptr_40[_blk];
                        uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) << 16);
                            (&_vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float _vec_load_15[8];
                {
                    const uint4* _vptr_41 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + (i_5 + 1) * split_stride) + 0);
                    uint4 _vld_41[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_41[_blk] = _vptr_41[_blk];
                        uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                            (&_vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float _vec_load_16[8];
                {
                    const uint4* _vptr_42 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + (i_5 + 1) * split_stride + 8) + 0);
                    uint4 _vld_42[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_42[_blk] = _vptr_42[_blk];
                        uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                            (&_vec_load_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_13 = 0; j_13 < 8; j_13++) {
                    acc[j_13] = __fmaf_rn(_vec_load_13[j_13], w_i_1, acc[j_13]);
                    acc[8 + j_13] = __fmaf_rn(_vec_load_14[j_13], w_i_1, acc[8 + j_13]);
                }
                #pragma unroll
                for (int j_14 = 0; j_14 < 8; j_14++) {
                    acc[j_14] = __fmaf_rn(_vec_load_15[j_14], w_n_1, acc[j_14]);
                    acc[8 + j_14] = __fmaf_rn(_vec_load_16[j_14], w_n_1, acc[8 + j_14]);
                }
            }
            if ((used & 1) != 0) {
                int i_last_1 = used - 1;
                float w_l_1 = approx_exp2(partial_lse[lse_base_r + i_last_1] - global_lse2);
                #pragma unroll
                for (int j8_1 = 0; j8_1 < 16; j8_1 += 8) {
                    float _vec_load_17[8];
                    {
                        const uint4* _vptr_43 = reinterpret_cast<const uint4*>(partial_o + (o_base_r + i_last_1 * split_stride + j8_1) + 0);
                        uint4 _vld_43[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_43[_blk] = _vptr_43[_blk];
                            uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                                (&_vec_load_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_15 = 0; j_15 < 8; j_15++) {
                        acc[j8_1 + j_15] = __fmaf_rn(_vec_load_17[j_15], w_l_1, acc[j8_1 + j_15]);
                    }
                }
            }
            unsigned int _packed_3[8];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                _packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
                _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 4], acc[_lp*2+1 + 4]));
                _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 6], acc[_lp*2+1 + 6]));
                _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
                _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 10], acc[_lp*2+1 + 10]));
                _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 12], acc[_lp*2+1 + 12]));
                _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 14], acc[_lp*2+1 + 14]));
                _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(o + (row[1] * 512 + lane_0 * 16))[0] = reinterpret_cast<int4*>(_packed_3)[0];
            reinterpret_cast<int4*>(o + (row[1] * 512 + lane_0 * 16 + 8))[0] = reinterpret_cast<int4*>(_packed_3 + (4))[0];
        }
    }
}

} // extern "C"
