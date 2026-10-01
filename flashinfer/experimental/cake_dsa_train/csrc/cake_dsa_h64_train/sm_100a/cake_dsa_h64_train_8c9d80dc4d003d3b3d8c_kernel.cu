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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_cake_dsa_h64_train_8c9d80dc4d003d3b3d8c(float* __restrict__ src_latent, float* __restrict__ src_rope, __nv_bfloat16* __restrict__ dst_latent, __nv_bfloat16* __restrict__ dst_rope, float* __restrict__ dst_latent_f32, float* __restrict__ dst_rope_f32, int latent_groups, int rope_groups, int out_f32)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int g = blockIdx.x * 256 + warp * 32 + lane;
    if (g < latent_groups) {
        float vals[32];
        float nat[32];
        long long base = (long long)g * 32;
        float _vec_load_0[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_latent + base + 0);
            _vec_load_0[0 + 0] = _v4.x;
            _vec_load_0[0 + 1] = _v4.y;
            _vec_load_0[0 + 2] = _v4.z;
            _vec_load_0[0 + 3] = _v4.w;
        }
        vals[0] = _vec_load_0[0];
        vals[1] = _vec_load_0[1];
        vals[2] = _vec_load_0[2];
        vals[3] = _vec_load_0[3];
        float _vec_load_1[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_latent + (base + 4) + 0);
            _vec_load_1[0 + 0] = _v4.x;
            _vec_load_1[0 + 1] = _v4.y;
            _vec_load_1[0 + 2] = _v4.z;
            _vec_load_1[0 + 3] = _v4.w;
        }
        vals[4] = _vec_load_1[0];
        vals[5] = _vec_load_1[1];
        vals[6] = _vec_load_1[2];
        vals[7] = _vec_load_1[3];
        float _vec_load_2[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_latent + (base + 8) + 0);
            _vec_load_2[0 + 0] = _v4.x;
            _vec_load_2[0 + 1] = _v4.y;
            _vec_load_2[0 + 2] = _v4.z;
            _vec_load_2[0 + 3] = _v4.w;
        }
        vals[8] = _vec_load_2[0];
        vals[9] = _vec_load_2[1];
        vals[10] = _vec_load_2[2];
        vals[11] = _vec_load_2[3];
        float _vec_load_3[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_latent + (base + 12) + 0);
            _vec_load_3[0 + 0] = _v4.x;
            _vec_load_3[0 + 1] = _v4.y;
            _vec_load_3[0 + 2] = _v4.z;
            _vec_load_3[0 + 3] = _v4.w;
        }
        vals[12] = _vec_load_3[0];
        vals[13] = _vec_load_3[1];
        vals[14] = _vec_load_3[2];
        vals[15] = _vec_load_3[3];
        float _vec_load_4[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_latent + (base + 16) + 0);
            _vec_load_4[0 + 0] = _v4.x;
            _vec_load_4[0 + 1] = _v4.y;
            _vec_load_4[0 + 2] = _v4.z;
            _vec_load_4[0 + 3] = _v4.w;
        }
        vals[16] = _vec_load_4[0];
        vals[17] = _vec_load_4[1];
        vals[18] = _vec_load_4[2];
        vals[19] = _vec_load_4[3];
        float _vec_load_5[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_latent + (base + 20) + 0);
            _vec_load_5[0 + 0] = _v4.x;
            _vec_load_5[0 + 1] = _v4.y;
            _vec_load_5[0 + 2] = _v4.z;
            _vec_load_5[0 + 3] = _v4.w;
        }
        vals[20] = _vec_load_5[0];
        vals[21] = _vec_load_5[1];
        vals[22] = _vec_load_5[2];
        vals[23] = _vec_load_5[3];
        float _vec_load_6[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_latent + (base + 24) + 0);
            _vec_load_6[0 + 0] = _v4.x;
            _vec_load_6[0 + 1] = _v4.y;
            _vec_load_6[0 + 2] = _v4.z;
            _vec_load_6[0 + 3] = _v4.w;
        }
        vals[24] = _vec_load_6[0];
        vals[25] = _vec_load_6[1];
        vals[26] = _vec_load_6[2];
        vals[27] = _vec_load_6[3];
        float _vec_load_7[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_latent + (base + 28) + 0);
            _vec_load_7[0 + 0] = _v4.x;
            _vec_load_7[0 + 1] = _v4.y;
            _vec_load_7[0 + 2] = _v4.z;
            _vec_load_7[0 + 3] = _v4.w;
        }
        vals[28] = _vec_load_7[0];
        vals[29] = _vec_load_7[1];
        vals[30] = _vec_load_7[2];
        vals[31] = _vec_load_7[3];
        nat[0] = vals[0];
        nat[8] = vals[1];
        nat[16] = vals[2];
        nat[24] = vals[3];
        nat[1] = vals[4];
        nat[9] = vals[5];
        nat[17] = vals[6];
        nat[25] = vals[7];
        nat[2] = vals[8];
        nat[10] = vals[9];
        nat[18] = vals[10];
        nat[26] = vals[11];
        nat[3] = vals[12];
        nat[11] = vals[13];
        nat[19] = vals[14];
        nat[27] = vals[15];
        nat[4] = vals[16];
        nat[12] = vals[17];
        nat[20] = vals[18];
        nat[28] = vals[19];
        nat[5] = vals[20];
        nat[13] = vals[21];
        nat[21] = vals[22];
        nat[29] = vals[23];
        nat[6] = vals[24];
        nat[14] = vals[25];
        nat[22] = vals[26];
        nat[30] = vals[27];
        nat[7] = vals[28];
        nat[15] = vals[29];
        nat[23] = vals[30];
        nat[31] = vals[31];
        if (out_f32 != 0) {
            {
                float4 _v4 = make_float4(nat[0 + 0], nat[0 + 1], nat[0 + 2], nat[0 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + base) = _v4;
            }
            {
                float4 _v4 = make_float4(nat[4 + 0], nat[4 + 1], nat[4 + 2], nat[4 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + base + 4) = _v4;
            }
            {
                float4 _v4 = make_float4(nat[8 + 0], nat[8 + 1], nat[8 + 2], nat[8 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + base + 8) = _v4;
            }
            {
                float4 _v4 = make_float4(nat[12 + 0], nat[12 + 1], nat[12 + 2], nat[12 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + base + 12) = _v4;
            }
            {
                float4 _v4 = make_float4(nat[16 + 0], nat[16 + 1], nat[16 + 2], nat[16 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + base + 16) = _v4;
            }
            {
                float4 _v4 = make_float4(nat[20 + 0], nat[20 + 1], nat[20 + 2], nat[20 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + base + 20) = _v4;
            }
            {
                float4 _v4 = make_float4(nat[24 + 0], nat[24 + 1], nat[24 + 2], nat[24 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + base + 24) = _v4;
            }
            {
                float4 _v4 = make_float4(nat[28 + 0], nat[28 + 1], nat[28 + 2], nat[28 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + base + 28) = _v4;
            }
        } else {
            {
                {
                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(nat[0 + 0], nat[0 + 1]);
                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(nat[0 + 2], nat[0 + 3]);
                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(nat[0 + 4], nat[0 + 5]);
                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(nat[0 + 6], nat[0 + 7]);
                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(nat[0 + 8], nat[0 + 9]);
                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(nat[0 + 10], nat[0 + 11]);
                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(nat[0 + 12], nat[0 + 13]);
                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(nat[0 + 14], nat[0 + 15]);
                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(&((__nv_bfloat16*)(dst_latent))[base + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                }
            }
            {
                {
                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(nat[16 + 0], nat[16 + 1]);
                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(nat[16 + 2], nat[16 + 3]);
                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(nat[16 + 4], nat[16 + 5]);
                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(nat[16 + 6], nat[16 + 7]);
                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(nat[16 + 8], nat[16 + 9]);
                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(nat[16 + 10], nat[16 + 11]);
                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(nat[16 + 12], nat[16 + 13]);
                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(nat[16 + 14], nat[16 + 15]);
                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(&((__nv_bfloat16*)(dst_latent))[base + 16 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                }
            }
        }
    } else if (g < latent_groups + rope_groups) {
        float rvals[32];
        float rnat[32];
        long long rbase = (long long)(g - latent_groups) * 32;
        float _vec_load_8[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + rbase + 0);
            _vec_load_8[0 + 0] = _v4.x;
            _vec_load_8[0 + 1] = _v4.y;
            _vec_load_8[0 + 2] = _v4.z;
            _vec_load_8[0 + 3] = _v4.w;
        }
        rvals[0] = _vec_load_8[0];
        rvals[1] = _vec_load_8[1];
        rvals[2] = _vec_load_8[2];
        rvals[3] = _vec_load_8[3];
        float _vec_load_9[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 4) + 0);
            _vec_load_9[0 + 0] = _v4.x;
            _vec_load_9[0 + 1] = _v4.y;
            _vec_load_9[0 + 2] = _v4.z;
            _vec_load_9[0 + 3] = _v4.w;
        }
        rvals[4] = _vec_load_9[0];
        rvals[5] = _vec_load_9[1];
        rvals[6] = _vec_load_9[2];
        rvals[7] = _vec_load_9[3];
        float _vec_load_10[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 8) + 0);
            _vec_load_10[0 + 0] = _v4.x;
            _vec_load_10[0 + 1] = _v4.y;
            _vec_load_10[0 + 2] = _v4.z;
            _vec_load_10[0 + 3] = _v4.w;
        }
        rvals[8] = _vec_load_10[0];
        rvals[9] = _vec_load_10[1];
        rvals[10] = _vec_load_10[2];
        rvals[11] = _vec_load_10[3];
        float _vec_load_11[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 12) + 0);
            _vec_load_11[0 + 0] = _v4.x;
            _vec_load_11[0 + 1] = _v4.y;
            _vec_load_11[0 + 2] = _v4.z;
            _vec_load_11[0 + 3] = _v4.w;
        }
        rvals[12] = _vec_load_11[0];
        rvals[13] = _vec_load_11[1];
        rvals[14] = _vec_load_11[2];
        rvals[15] = _vec_load_11[3];
        float _vec_load_12[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 16) + 0);
            _vec_load_12[0 + 0] = _v4.x;
            _vec_load_12[0 + 1] = _v4.y;
            _vec_load_12[0 + 2] = _v4.z;
            _vec_load_12[0 + 3] = _v4.w;
        }
        rvals[16] = _vec_load_12[0];
        rvals[17] = _vec_load_12[1];
        rvals[18] = _vec_load_12[2];
        rvals[19] = _vec_load_12[3];
        float _vec_load_13[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 20) + 0);
            _vec_load_13[0 + 0] = _v4.x;
            _vec_load_13[0 + 1] = _v4.y;
            _vec_load_13[0 + 2] = _v4.z;
            _vec_load_13[0 + 3] = _v4.w;
        }
        rvals[20] = _vec_load_13[0];
        rvals[21] = _vec_load_13[1];
        rvals[22] = _vec_load_13[2];
        rvals[23] = _vec_load_13[3];
        float _vec_load_14[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 24) + 0);
            _vec_load_14[0 + 0] = _v4.x;
            _vec_load_14[0 + 1] = _v4.y;
            _vec_load_14[0 + 2] = _v4.z;
            _vec_load_14[0 + 3] = _v4.w;
        }
        rvals[24] = _vec_load_14[0];
        rvals[25] = _vec_load_14[1];
        rvals[26] = _vec_load_14[2];
        rvals[27] = _vec_load_14[3];
        float _vec_load_15[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 28) + 0);
            _vec_load_15[0 + 0] = _v4.x;
            _vec_load_15[0 + 1] = _v4.y;
            _vec_load_15[0 + 2] = _v4.z;
            _vec_load_15[0 + 3] = _v4.w;
        }
        rvals[28] = _vec_load_15[0];
        rvals[29] = _vec_load_15[1];
        rvals[30] = _vec_load_15[2];
        rvals[31] = _vec_load_15[3];
        rnat[0] = rvals[0];
        rnat[16] = rvals[16];
        rnat[8] = rvals[1];
        rnat[24] = rvals[17];
        rnat[1] = rvals[2];
        rnat[17] = rvals[18];
        rnat[9] = rvals[3];
        rnat[25] = rvals[19];
        rnat[2] = rvals[4];
        rnat[18] = rvals[20];
        rnat[10] = rvals[5];
        rnat[26] = rvals[21];
        rnat[3] = rvals[6];
        rnat[19] = rvals[22];
        rnat[11] = rvals[7];
        rnat[27] = rvals[23];
        rnat[4] = rvals[8];
        rnat[20] = rvals[24];
        rnat[12] = rvals[9];
        rnat[28] = rvals[25];
        rnat[5] = rvals[10];
        rnat[21] = rvals[26];
        rnat[13] = rvals[11];
        rnat[29] = rvals[27];
        rnat[6] = rvals[12];
        rnat[22] = rvals[28];
        rnat[14] = rvals[13];
        rnat[30] = rvals[29];
        rnat[7] = rvals[14];
        rnat[23] = rvals[30];
        rnat[15] = rvals[15];
        rnat[31] = rvals[31];
        if (out_f32 != 0) {
            {
                float4 _v4 = make_float4(rnat[0 + 0], rnat[0 + 1], rnat[0 + 2], rnat[0 + 3]);
                *reinterpret_cast<float4*>(dst_rope_f32 + rbase) = _v4;
            }
            {
                float4 _v4 = make_float4(rnat[4 + 0], rnat[4 + 1], rnat[4 + 2], rnat[4 + 3]);
                *reinterpret_cast<float4*>(dst_rope_f32 + rbase + 4) = _v4;
            }
            {
                float4 _v4 = make_float4(rnat[8 + 0], rnat[8 + 1], rnat[8 + 2], rnat[8 + 3]);
                *reinterpret_cast<float4*>(dst_rope_f32 + rbase + 8) = _v4;
            }
            {
                float4 _v4 = make_float4(rnat[12 + 0], rnat[12 + 1], rnat[12 + 2], rnat[12 + 3]);
                *reinterpret_cast<float4*>(dst_rope_f32 + rbase + 12) = _v4;
            }
            {
                float4 _v4 = make_float4(rnat[16 + 0], rnat[16 + 1], rnat[16 + 2], rnat[16 + 3]);
                *reinterpret_cast<float4*>(dst_rope_f32 + rbase + 16) = _v4;
            }
            {
                float4 _v4 = make_float4(rnat[20 + 0], rnat[20 + 1], rnat[20 + 2], rnat[20 + 3]);
                *reinterpret_cast<float4*>(dst_rope_f32 + rbase + 20) = _v4;
            }
            {
                float4 _v4 = make_float4(rnat[24 + 0], rnat[24 + 1], rnat[24 + 2], rnat[24 + 3]);
                *reinterpret_cast<float4*>(dst_rope_f32 + rbase + 24) = _v4;
            }
            {
                float4 _v4 = make_float4(rnat[28 + 0], rnat[28 + 1], rnat[28 + 2], rnat[28 + 3]);
                *reinterpret_cast<float4*>(dst_rope_f32 + rbase + 28) = _v4;
            }
        } else {
            {
                {
                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(rnat[0 + 0], rnat[0 + 1]);
                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(rnat[0 + 2], rnat[0 + 3]);
                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(rnat[0 + 4], rnat[0 + 5]);
                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(rnat[0 + 6], rnat[0 + 7]);
                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(rnat[0 + 8], rnat[0 + 9]);
                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(rnat[0 + 10], rnat[0 + 11]);
                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(rnat[0 + 12], rnat[0 + 13]);
                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(rnat[0 + 14], rnat[0 + 15]);
                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(&((__nv_bfloat16*)(dst_rope))[rbase + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                }
            }
            {
                {
                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(rnat[16 + 0], rnat[16 + 1]);
                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(rnat[16 + 2], rnat[16 + 3]);
                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(rnat[16 + 4], rnat[16 + 5]);
                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(rnat[16 + 6], rnat[16 + 7]);
                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(rnat[16 + 8], rnat[16 + 9]);
                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(rnat[16 + 10], rnat[16 + 11]);
                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(rnat[16 + 12], rnat[16 + 13]);
                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(rnat[16 + 14], rnat[16 + 15]);
                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(&((__nv_bfloat16*)(dst_rope))[rbase + 16 + 0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                }
            }
        }
    }
}

} // extern "C"
