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

#include <math_constants.h>


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kda_det_train_f793454fc9c70b991661(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, float* __restrict__ rstd, __nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, int num_tokens, int num_heads, int q_tstride, int q_hstride, int k_tstride, int k_hstride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int h = blockIdx.y;
    int c = tid & 15;
    int t0 = blockIdx.x * 64 + (tid >> 4);
    float x4[32];
    int _min_0 = ((t0) < (num_tokens - 1) ? (t0) : (num_tokens - 1));
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(q + ((long long)_min_0 * (long long)q_tstride + (long long)h * (long long)q_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_1[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_1[_blk] = _vptr_1[_blk];
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&x4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_2 = ((t0 + 16) < (num_tokens - 1) ? (t0 + 16) : (num_tokens - 1));
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(q + ((long long)_min_2 * (long long)q_tstride + (long long)h * (long long)q_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_3[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_3[_blk] = _vptr_3[_blk];
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&x4[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_4 = ((t0 + 32) < (num_tokens - 1) ? (t0 + 32) : (num_tokens - 1));
    {
        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(q + ((long long)_min_4 * (long long)q_tstride + (long long)h * (long long)q_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_5[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_5[_blk] = _vptr_5[_blk];
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&x4[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_6 = ((t0 + 48) < (num_tokens - 1) ? (t0 + 48) : (num_tokens - 1));
    {
        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(q + ((long long)_min_6 * (long long)q_tstride + (long long)h * (long long)q_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_7[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_7[_blk] = _vptr_7[_blk];
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                (&x4[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
            }
        }
    }
    float r[4];
    if (t0 < num_tokens) {
        {
            r[0] = *reinterpret_cast<const float*>(rstd + ((long long)t0 * (long long)num_heads + (long long)h) + 0);
        }
        float y[8];
        y[0] = mul_f32_rn_noftz(x4[0], r[0]);
        y[1] = mul_f32_rn_noftz(x4[1], r[0]);
        y[2] = mul_f32_rn_noftz(x4[2], r[0]);
        y[3] = mul_f32_rn_noftz(x4[3], r[0]);
        y[4] = mul_f32_rn_noftz(x4[4], r[0]);
        y[5] = mul_f32_rn_noftz(x4[5], r[0]);
        y[6] = mul_f32_rn_noftz(x4[6], r[0]);
        y[7] = mul_f32_rn_noftz(x4[7], r[0]);
        unsigned int yb[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
            yb[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)t0 * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb)[0];
    }
    if (t0 + 16 < num_tokens) {
        {
            r[1] = *reinterpret_cast<const float*>(rstd + ((long long)(t0 + 16) * (long long)num_heads + (long long)h) + 0);
        }
        float y_1[8];
        y_1[0] = mul_f32_rn_noftz(x4[8], r[1]);
        y_1[1] = mul_f32_rn_noftz(x4[9], r[1]);
        y_1[2] = mul_f32_rn_noftz(x4[10], r[1]);
        y_1[3] = mul_f32_rn_noftz(x4[11], r[1]);
        y_1[4] = mul_f32_rn_noftz(x4[12], r[1]);
        y_1[5] = mul_f32_rn_noftz(x4[13], r[1]);
        y_1[6] = mul_f32_rn_noftz(x4[14], r[1]);
        y_1[7] = mul_f32_rn_noftz(x4[15], r[1]);
        unsigned int yb_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_1[_lp*2 + 0], y_1[_lp*2+1 + 0]));
            yb_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(t0 + 16) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_1)[0];
    }
    if (t0 + 32 < num_tokens) {
        {
            r[2] = *reinterpret_cast<const float*>(rstd + ((long long)(t0 + 32) * (long long)num_heads + (long long)h) + 0);
        }
        float y_2[8];
        y_2[0] = mul_f32_rn_noftz(x4[16], r[2]);
        y_2[1] = mul_f32_rn_noftz(x4[17], r[2]);
        y_2[2] = mul_f32_rn_noftz(x4[18], r[2]);
        y_2[3] = mul_f32_rn_noftz(x4[19], r[2]);
        y_2[4] = mul_f32_rn_noftz(x4[20], r[2]);
        y_2[5] = mul_f32_rn_noftz(x4[21], r[2]);
        y_2[6] = mul_f32_rn_noftz(x4[22], r[2]);
        y_2[7] = mul_f32_rn_noftz(x4[23], r[2]);
        unsigned int yb_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_2[_lp*2 + 0], y_2[_lp*2+1 + 0]));
            yb_2[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(t0 + 32) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_2)[0];
    }
    if (t0 + 48 < num_tokens) {
        {
            r[3] = *reinterpret_cast<const float*>(rstd + ((long long)(t0 + 48) * (long long)num_heads + (long long)h) + 0);
        }
        float y_3[8];
        y_3[0] = mul_f32_rn_noftz(x4[24], r[3]);
        y_3[1] = mul_f32_rn_noftz(x4[25], r[3]);
        y_3[2] = mul_f32_rn_noftz(x4[26], r[3]);
        y_3[3] = mul_f32_rn_noftz(x4[27], r[3]);
        y_3[4] = mul_f32_rn_noftz(x4[28], r[3]);
        y_3[5] = mul_f32_rn_noftz(x4[29], r[3]);
        y_3[6] = mul_f32_rn_noftz(x4[30], r[3]);
        y_3[7] = mul_f32_rn_noftz(x4[31], r[3]);
        unsigned int yb_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_3[_lp*2 + 0], y_3[_lp*2+1 + 0]));
            yb_3[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(t0 + 48) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_3)[0];
    }
    float x4_0[32];
    int _min_8 = ((t0) < (num_tokens - 1) ? (t0) : (num_tokens - 1));
    {
        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(k + ((long long)_min_8 * (long long)k_tstride + (long long)h * (long long)k_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_9[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_9[_blk] = _vptr_9[_blk];
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                (&x4_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_10 = ((t0 + 16) < (num_tokens - 1) ? (t0 + 16) : (num_tokens - 1));
    {
        const uint4* _vptr_11 = reinterpret_cast<const uint4*>(k + ((long long)_min_10 * (long long)k_tstride + (long long)h * (long long)k_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_11[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_11[_blk] = _vptr_11[_blk];
            uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4_0[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                (&x4_0[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_12 = ((t0 + 32) < (num_tokens - 1) ? (t0 + 32) : (num_tokens - 1));
    {
        const uint4* _vptr_13 = reinterpret_cast<const uint4*>(k + ((long long)_min_12 * (long long)k_tstride + (long long)h * (long long)k_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_13[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_13[_blk] = _vptr_13[_blk];
            uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4_0[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                (&x4_0[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_14 = ((t0 + 48) < (num_tokens - 1) ? (t0 + 48) : (num_tokens - 1));
    {
        const uint4* _vptr_15 = reinterpret_cast<const uint4*>(k + ((long long)_min_14 * (long long)k_tstride + (long long)h * (long long)k_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_15[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_15[_blk] = _vptr_15[_blk];
            uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4_0[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                (&x4_0[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
            }
        }
    }
    float r_0[4];
    if (t0 < num_tokens) {
        {
            r_0[0] = *reinterpret_cast<const float*>(rstd + ((long long)(num_tokens + t0) * (long long)num_heads + (long long)h) + 0);
        }
        float y_4[8];
        y_4[0] = mul_f32_rn_noftz(x4_0[0], r_0[0]);
        y_4[1] = mul_f32_rn_noftz(x4_0[1], r_0[0]);
        y_4[2] = mul_f32_rn_noftz(x4_0[2], r_0[0]);
        y_4[3] = mul_f32_rn_noftz(x4_0[3], r_0[0]);
        y_4[4] = mul_f32_rn_noftz(x4_0[4], r_0[0]);
        y_4[5] = mul_f32_rn_noftz(x4_0[5], r_0[0]);
        y_4[6] = mul_f32_rn_noftz(x4_0[6], r_0[0]);
        y_4[7] = mul_f32_rn_noftz(x4_0[7], r_0[0]);
        unsigned int yb_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_4[_lp*2 + 0], y_4[_lp*2+1 + 0]));
            yb_4[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)t0 * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_4)[0];
    }
    if (t0 + 16 < num_tokens) {
        {
            r_0[1] = *reinterpret_cast<const float*>(rstd + ((long long)(num_tokens + t0 + 16) * (long long)num_heads + (long long)h) + 0);
        }
        float y_5[8];
        y_5[0] = mul_f32_rn_noftz(x4_0[8], r_0[1]);
        y_5[1] = mul_f32_rn_noftz(x4_0[9], r_0[1]);
        y_5[2] = mul_f32_rn_noftz(x4_0[10], r_0[1]);
        y_5[3] = mul_f32_rn_noftz(x4_0[11], r_0[1]);
        y_5[4] = mul_f32_rn_noftz(x4_0[12], r_0[1]);
        y_5[5] = mul_f32_rn_noftz(x4_0[13], r_0[1]);
        y_5[6] = mul_f32_rn_noftz(x4_0[14], r_0[1]);
        y_5[7] = mul_f32_rn_noftz(x4_0[15], r_0[1]);
        unsigned int yb_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_5[_lp*2 + 0], y_5[_lp*2+1 + 0]));
            yb_5[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(t0 + 16) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_5)[0];
    }
    if (t0 + 32 < num_tokens) {
        {
            r_0[2] = *reinterpret_cast<const float*>(rstd + ((long long)(num_tokens + t0 + 32) * (long long)num_heads + (long long)h) + 0);
        }
        float y_6[8];
        y_6[0] = mul_f32_rn_noftz(x4_0[16], r_0[2]);
        y_6[1] = mul_f32_rn_noftz(x4_0[17], r_0[2]);
        y_6[2] = mul_f32_rn_noftz(x4_0[18], r_0[2]);
        y_6[3] = mul_f32_rn_noftz(x4_0[19], r_0[2]);
        y_6[4] = mul_f32_rn_noftz(x4_0[20], r_0[2]);
        y_6[5] = mul_f32_rn_noftz(x4_0[21], r_0[2]);
        y_6[6] = mul_f32_rn_noftz(x4_0[22], r_0[2]);
        y_6[7] = mul_f32_rn_noftz(x4_0[23], r_0[2]);
        unsigned int yb_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_6[_lp*2 + 0], y_6[_lp*2+1 + 0]));
            yb_6[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(t0 + 32) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_6)[0];
    }
    if (t0 + 48 < num_tokens) {
        {
            r_0[3] = *reinterpret_cast<const float*>(rstd + ((long long)(num_tokens + t0 + 48) * (long long)num_heads + (long long)h) + 0);
        }
        float y_7[8];
        y_7[0] = mul_f32_rn_noftz(x4_0[24], r_0[3]);
        y_7[1] = mul_f32_rn_noftz(x4_0[25], r_0[3]);
        y_7[2] = mul_f32_rn_noftz(x4_0[26], r_0[3]);
        y_7[3] = mul_f32_rn_noftz(x4_0[27], r_0[3]);
        y_7[4] = mul_f32_rn_noftz(x4_0[28], r_0[3]);
        y_7[5] = mul_f32_rn_noftz(x4_0[29], r_0[3]);
        y_7[6] = mul_f32_rn_noftz(x4_0[30], r_0[3]);
        y_7[7] = mul_f32_rn_noftz(x4_0[31], r_0[3]);
        unsigned int yb_7[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_7[_lp*2 + 0], y_7[_lp*2+1 + 0]));
            yb_7[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(t0 + 48) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_7)[0];
    }
}

} // extern "C"
