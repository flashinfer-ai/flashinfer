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
#define THREADS 256

#include <math_constants.h>


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gdn_det_train_dd20d92fd03ad2cf563d(__nv_bfloat16* __restrict__ q_raw, __nv_bfloat16* __restrict__ k_raw, float* __restrict__ q_rstd, float* __restrict__ k_rstd, __nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, int num_qk_heads, int q_raw_tstride, int k_raw_tstride, int rstd_tstride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int bos = chunk_bos[blockIdx.x];
    int clen = chunk_len[blockIdx.x];
    int h = blockIdx.y;
    int c = tid & 15;
    int s = tid >> 4;
    if (s < clen) {
        float xv[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q_raw + ((long long)(bos + s) * (long long)q_raw_tstride + (long long)(h * 128 + c * 8)) + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&xv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        float rv[1];
        {
            rv[0] = *reinterpret_cast<const float*>(q_rstd + ((long long)(bos + s) * (long long)rstd_tstride + (long long)h) + 0);
        }
        float yv[8];
        yv[0] = mul_f32_rn_noftz(xv[0], rv[0]);
        yv[1] = mul_f32_rn_noftz(xv[1], rv[0]);
        yv[2] = mul_f32_rn_noftz(xv[2], rv[0]);
        yv[3] = mul_f32_rn_noftz(xv[3], rv[0]);
        yv[4] = mul_f32_rn_noftz(xv[4], rv[0]);
        yv[5] = mul_f32_rn_noftz(xv[5], rv[0]);
        yv[6] = mul_f32_rn_noftz(xv[6], rv[0]);
        yv[7] = mul_f32_rn_noftz(xv[7], rv[0]);
        unsigned int yb[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv[_lp*2 + 0], yv[_lp*2+1 + 0]));
            yb[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(bos + s) * (long long)num_qk_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb)[0];
        float xv_0[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(k_raw + ((long long)(bos + s) * (long long)k_raw_tstride + (long long)(h * 128 + c * 8)) + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xv_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&xv_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        float rv_0[1];
        {
            rv_0[0] = *reinterpret_cast<const float*>(k_rstd + ((long long)(bos + s) * (long long)rstd_tstride + (long long)h) + 0);
        }
        float yv_0[8];
        yv_0[0] = mul_f32_rn_noftz(xv_0[0], rv_0[0]);
        yv_0[1] = mul_f32_rn_noftz(xv_0[1], rv_0[0]);
        yv_0[2] = mul_f32_rn_noftz(xv_0[2], rv_0[0]);
        yv_0[3] = mul_f32_rn_noftz(xv_0[3], rv_0[0]);
        yv_0[4] = mul_f32_rn_noftz(xv_0[4], rv_0[0]);
        yv_0[5] = mul_f32_rn_noftz(xv_0[5], rv_0[0]);
        yv_0[6] = mul_f32_rn_noftz(xv_0[6], rv_0[0]);
        yv_0[7] = mul_f32_rn_noftz(xv_0[7], rv_0[0]);
        unsigned int yb_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_0[_lp*2 + 0], yv_0[_lp*2+1 + 0]));
            yb_0[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(bos + s) * (long long)num_qk_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_0)[0];
    }
    if (s + 16 < clen) {
        float xv_1[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(q_raw + ((long long)(bos + s + 16) * (long long)q_raw_tstride + (long long)(h * 128 + c * 8)) + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xv_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&xv_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
        }
        float rv_1[1];
        {
            rv_1[0] = *reinterpret_cast<const float*>(q_rstd + ((long long)(bos + s + 16) * (long long)rstd_tstride + (long long)h) + 0);
        }
        float yv_1[8];
        yv_1[0] = mul_f32_rn_noftz(xv_1[0], rv_1[0]);
        yv_1[1] = mul_f32_rn_noftz(xv_1[1], rv_1[0]);
        yv_1[2] = mul_f32_rn_noftz(xv_1[2], rv_1[0]);
        yv_1[3] = mul_f32_rn_noftz(xv_1[3], rv_1[0]);
        yv_1[4] = mul_f32_rn_noftz(xv_1[4], rv_1[0]);
        yv_1[5] = mul_f32_rn_noftz(xv_1[5], rv_1[0]);
        yv_1[6] = mul_f32_rn_noftz(xv_1[6], rv_1[0]);
        yv_1[7] = mul_f32_rn_noftz(xv_1[7], rv_1[0]);
        unsigned int yb_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_1[_lp*2 + 0], yv_1[_lp*2+1 + 0]));
            yb_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(bos + s + 16) * (long long)num_qk_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_1)[0];
        float xv_1_1[8];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(k_raw + ((long long)(bos + s + 16) * (long long)k_raw_tstride + (long long)(h * 128 + c * 8)) + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xv_1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&xv_1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
        }
        float rv_1_1[1];
        {
            rv_1_1[0] = *reinterpret_cast<const float*>(k_rstd + ((long long)(bos + s + 16) * (long long)rstd_tstride + (long long)h) + 0);
        }
        float yv_1_1[8];
        yv_1_1[0] = mul_f32_rn_noftz(xv_1_1[0], rv_1_1[0]);
        yv_1_1[1] = mul_f32_rn_noftz(xv_1_1[1], rv_1_1[0]);
        yv_1_1[2] = mul_f32_rn_noftz(xv_1_1[2], rv_1_1[0]);
        yv_1_1[3] = mul_f32_rn_noftz(xv_1_1[3], rv_1_1[0]);
        yv_1_1[4] = mul_f32_rn_noftz(xv_1_1[4], rv_1_1[0]);
        yv_1_1[5] = mul_f32_rn_noftz(xv_1_1[5], rv_1_1[0]);
        yv_1_1[6] = mul_f32_rn_noftz(xv_1_1[6], rv_1_1[0]);
        yv_1_1[7] = mul_f32_rn_noftz(xv_1_1[7], rv_1_1[0]);
        unsigned int yb_1_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_1_1[_lp*2 + 0], yv_1_1[_lp*2+1 + 0]));
            yb_1_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(bos + s + 16) * (long long)num_qk_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_1_1)[0];
    }
    if (s + 32 < clen) {
        float xv_2[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(q_raw + ((long long)(bos + s + 32) * (long long)q_raw_tstride + (long long)(h * 128 + c * 8)) + 0);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xv_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&xv_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
        }
        float rv_2[1];
        {
            rv_2[0] = *reinterpret_cast<const float*>(q_rstd + ((long long)(bos + s + 32) * (long long)rstd_tstride + (long long)h) + 0);
        }
        float yv_2[8];
        yv_2[0] = mul_f32_rn_noftz(xv_2[0], rv_2[0]);
        yv_2[1] = mul_f32_rn_noftz(xv_2[1], rv_2[0]);
        yv_2[2] = mul_f32_rn_noftz(xv_2[2], rv_2[0]);
        yv_2[3] = mul_f32_rn_noftz(xv_2[3], rv_2[0]);
        yv_2[4] = mul_f32_rn_noftz(xv_2[4], rv_2[0]);
        yv_2[5] = mul_f32_rn_noftz(xv_2[5], rv_2[0]);
        yv_2[6] = mul_f32_rn_noftz(xv_2[6], rv_2[0]);
        yv_2[7] = mul_f32_rn_noftz(xv_2[7], rv_2[0]);
        unsigned int yb_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_2[_lp*2 + 0], yv_2[_lp*2+1 + 0]));
            yb_2[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(bos + s + 32) * (long long)num_qk_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_2)[0];
        float xv_2_1[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(k_raw + ((long long)(bos + s + 32) * (long long)k_raw_tstride + (long long)(h * 128 + c * 8)) + 0);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xv_2_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&xv_2_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
        }
        float rv_2_1[1];
        {
            rv_2_1[0] = *reinterpret_cast<const float*>(k_rstd + ((long long)(bos + s + 32) * (long long)rstd_tstride + (long long)h) + 0);
        }
        float yv_2_1[8];
        yv_2_1[0] = mul_f32_rn_noftz(xv_2_1[0], rv_2_1[0]);
        yv_2_1[1] = mul_f32_rn_noftz(xv_2_1[1], rv_2_1[0]);
        yv_2_1[2] = mul_f32_rn_noftz(xv_2_1[2], rv_2_1[0]);
        yv_2_1[3] = mul_f32_rn_noftz(xv_2_1[3], rv_2_1[0]);
        yv_2_1[4] = mul_f32_rn_noftz(xv_2_1[4], rv_2_1[0]);
        yv_2_1[5] = mul_f32_rn_noftz(xv_2_1[5], rv_2_1[0]);
        yv_2_1[6] = mul_f32_rn_noftz(xv_2_1[6], rv_2_1[0]);
        yv_2_1[7] = mul_f32_rn_noftz(xv_2_1[7], rv_2_1[0]);
        unsigned int yb_2_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_2_1[_lp*2 + 0], yv_2_1[_lp*2+1 + 0]));
            yb_2_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(bos + s + 32) * (long long)num_qk_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_2_1)[0];
    }
    if (s + 48 < clen) {
        float xv_3[8];
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(q_raw + ((long long)(bos + s + 48) * (long long)q_raw_tstride + (long long)(h * 128 + c * 8)) + 0);
            uint4 _vld_6[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_6[_blk] = _vptr_6[_blk];
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xv_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                    (&xv_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                }
            }
        }
        float rv_3[1];
        {
            rv_3[0] = *reinterpret_cast<const float*>(q_rstd + ((long long)(bos + s + 48) * (long long)rstd_tstride + (long long)h) + 0);
        }
        float yv_3[8];
        yv_3[0] = mul_f32_rn_noftz(xv_3[0], rv_3[0]);
        yv_3[1] = mul_f32_rn_noftz(xv_3[1], rv_3[0]);
        yv_3[2] = mul_f32_rn_noftz(xv_3[2], rv_3[0]);
        yv_3[3] = mul_f32_rn_noftz(xv_3[3], rv_3[0]);
        yv_3[4] = mul_f32_rn_noftz(xv_3[4], rv_3[0]);
        yv_3[5] = mul_f32_rn_noftz(xv_3[5], rv_3[0]);
        yv_3[6] = mul_f32_rn_noftz(xv_3[6], rv_3[0]);
        yv_3[7] = mul_f32_rn_noftz(xv_3[7], rv_3[0]);
        unsigned int yb_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_3[_lp*2 + 0], yv_3[_lp*2+1 + 0]));
            yb_3[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(bos + s + 48) * (long long)num_qk_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_3)[0];
        float xv_3_1[8];
        {
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(k_raw + ((long long)(bos + s + 48) * (long long)k_raw_tstride + (long long)(h * 128 + c * 8)) + 0);
            uint4 _vld_7[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_7[_blk] = _vptr_7[_blk];
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xv_3_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                    (&xv_3_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                }
            }
        }
        float rv_3_1[1];
        {
            rv_3_1[0] = *reinterpret_cast<const float*>(k_rstd + ((long long)(bos + s + 48) * (long long)rstd_tstride + (long long)h) + 0);
        }
        float yv_3_1[8];
        yv_3_1[0] = mul_f32_rn_noftz(xv_3_1[0], rv_3_1[0]);
        yv_3_1[1] = mul_f32_rn_noftz(xv_3_1[1], rv_3_1[0]);
        yv_3_1[2] = mul_f32_rn_noftz(xv_3_1[2], rv_3_1[0]);
        yv_3_1[3] = mul_f32_rn_noftz(xv_3_1[3], rv_3_1[0]);
        yv_3_1[4] = mul_f32_rn_noftz(xv_3_1[4], rv_3_1[0]);
        yv_3_1[5] = mul_f32_rn_noftz(xv_3_1[5], rv_3_1[0]);
        yv_3_1[6] = mul_f32_rn_noftz(xv_3_1[6], rv_3_1[0]);
        yv_3_1[7] = mul_f32_rn_noftz(xv_3_1[7], rv_3_1[0]);
        unsigned int yb_3_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_3_1[_lp*2 + 0], yv_3_1[_lp*2+1 + 0]));
            yb_3_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(bos + s + 48) * (long long)num_qk_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_3_1)[0];
    }
}

} // extern "C"
