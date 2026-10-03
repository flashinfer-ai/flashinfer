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
kernel_cake_dsa_h64_train_8528ac191367cda50727(float* __restrict__ src_latent, float* __restrict__ src_rope, __nv_bfloat16* __restrict__ dst_latent, __nv_bfloat16* __restrict__ dst_rope, float* __restrict__ dst_latent_f32, float* __restrict__ dst_rope_f32, int latent_groups, int rope_groups, int out_f32, float* __restrict__ dst_packed, int dst_row_stride, int* __restrict__ dst_map, int has_dst_map, int accumulate)
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
        if (accumulate != 0) {
            float lnz = 1.0f;
            {
                float _fabs_0 = fabsf(vals[0]);
                lnz = _fabs_0;
                float _fabs_1 = fabsf(vals[1]);
                lnz = lnz + _fabs_1;
                float _fabs_2 = fabsf(vals[2]);
                lnz = lnz + _fabs_2;
                float _fabs_3 = fabsf(vals[3]);
                lnz = lnz + _fabs_3;
                float _fabs_4 = fabsf(vals[4]);
                lnz = lnz + _fabs_4;
                float _fabs_5 = fabsf(vals[5]);
                lnz = lnz + _fabs_5;
                float _fabs_6 = fabsf(vals[6]);
                lnz = lnz + _fabs_6;
                float _fabs_7 = fabsf(vals[7]);
                lnz = lnz + _fabs_7;
                float _fabs_8 = fabsf(vals[8]);
                lnz = lnz + _fabs_8;
                float _fabs_9 = fabsf(vals[9]);
                lnz = lnz + _fabs_9;
                float _fabs_10 = fabsf(vals[10]);
                lnz = lnz + _fabs_10;
                float _fabs_11 = fabsf(vals[11]);
                lnz = lnz + _fabs_11;
                float _fabs_12 = fabsf(vals[12]);
                lnz = lnz + _fabs_12;
                float _fabs_13 = fabsf(vals[13]);
                lnz = lnz + _fabs_13;
                float _fabs_14 = fabsf(vals[14]);
                lnz = lnz + _fabs_14;
                float _fabs_15 = fabsf(vals[15]);
                lnz = lnz + _fabs_15;
                float _fabs_16 = fabsf(vals[16]);
                lnz = lnz + _fabs_16;
                float _fabs_17 = fabsf(vals[17]);
                lnz = lnz + _fabs_17;
                float _fabs_18 = fabsf(vals[18]);
                lnz = lnz + _fabs_18;
                float _fabs_19 = fabsf(vals[19]);
                lnz = lnz + _fabs_19;
                float _fabs_20 = fabsf(vals[20]);
                lnz = lnz + _fabs_20;
                float _fabs_21 = fabsf(vals[21]);
                lnz = lnz + _fabs_21;
                float _fabs_22 = fabsf(vals[22]);
                lnz = lnz + _fabs_22;
                float _fabs_23 = fabsf(vals[23]);
                lnz = lnz + _fabs_23;
                float _fabs_24 = fabsf(vals[24]);
                lnz = lnz + _fabs_24;
                float _fabs_25 = fabsf(vals[25]);
                lnz = lnz + _fabs_25;
                float _fabs_26 = fabsf(vals[26]);
                lnz = lnz + _fabs_26;
                float _fabs_27 = fabsf(vals[27]);
                lnz = lnz + _fabs_27;
                float _fabs_28 = fabsf(vals[28]);
                lnz = lnz + _fabs_28;
                float _fabs_29 = fabsf(vals[29]);
                lnz = lnz + _fabs_29;
                float _fabs_30 = fabsf(vals[30]);
                lnz = lnz + _fabs_30;
                float _fabs_31 = fabsf(vals[31]);
                lnz = lnz + _fabs_31;
            }
            if (lnz != 0.0f) {
                int lrow = g / 16;
                if (has_dst_map != 0) {
                    lrow = dst_map[(long long)lrow];
                }
                long long ldst = (long long)lrow * (long long)dst_row_stride + (long long)(g % 16) * 32;
                if (has_dst_map != 0) {
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[ldst])), "f"(nat[0]), "f"(nat[1]), "f"(nat[2]), "f"(nat[3]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[ldst + 4])), "f"(nat[4]), "f"(nat[5]), "f"(nat[6]), "f"(nat[7]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[ldst + 8])), "f"(nat[8]), "f"(nat[9]), "f"(nat[10]), "f"(nat[11]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[ldst + 12])), "f"(nat[12]), "f"(nat[13]), "f"(nat[14]), "f"(nat[15]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[ldst + 16])), "f"(nat[16]), "f"(nat[17]), "f"(nat[18]), "f"(nat[19]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[ldst + 20])), "f"(nat[20]), "f"(nat[21]), "f"(nat[22]), "f"(nat[23]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[ldst + 24])), "f"(nat[24]), "f"(nat[25]), "f"(nat[26]), "f"(nat[27]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[ldst + 28])), "f"(nat[28]), "f"(nat[29]), "f"(nat[30]), "f"(nat[31]) : "memory");
                } else {
                    float _vec_load_8[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ldst + 0);
                        _vec_load_8[0 + 0] = _v4.x;
                        _vec_load_8[0 + 1] = _v4.y;
                        _vec_load_8[0 + 2] = _v4.z;
                        _vec_load_8[0 + 3] = _v4.w;
                    }
                    nat[0] = nat[0] + _vec_load_8[0];
                    nat[1] = nat[1] + _vec_load_8[1];
                    nat[2] = nat[2] + _vec_load_8[2];
                    nat[3] = nat[3] + _vec_load_8[3];
                    float _vec_load_9[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (ldst + 4) + 0);
                        _vec_load_9[0 + 0] = _v4.x;
                        _vec_load_9[0 + 1] = _v4.y;
                        _vec_load_9[0 + 2] = _v4.z;
                        _vec_load_9[0 + 3] = _v4.w;
                    }
                    nat[4] = nat[4] + _vec_load_9[0];
                    nat[5] = nat[5] + _vec_load_9[1];
                    nat[6] = nat[6] + _vec_load_9[2];
                    nat[7] = nat[7] + _vec_load_9[3];
                    float _vec_load_10[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (ldst + 8) + 0);
                        _vec_load_10[0 + 0] = _v4.x;
                        _vec_load_10[0 + 1] = _v4.y;
                        _vec_load_10[0 + 2] = _v4.z;
                        _vec_load_10[0 + 3] = _v4.w;
                    }
                    nat[8] = nat[8] + _vec_load_10[0];
                    nat[9] = nat[9] + _vec_load_10[1];
                    nat[10] = nat[10] + _vec_load_10[2];
                    nat[11] = nat[11] + _vec_load_10[3];
                    float _vec_load_11[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (ldst + 12) + 0);
                        _vec_load_11[0 + 0] = _v4.x;
                        _vec_load_11[0 + 1] = _v4.y;
                        _vec_load_11[0 + 2] = _v4.z;
                        _vec_load_11[0 + 3] = _v4.w;
                    }
                    nat[12] = nat[12] + _vec_load_11[0];
                    nat[13] = nat[13] + _vec_load_11[1];
                    nat[14] = nat[14] + _vec_load_11[2];
                    nat[15] = nat[15] + _vec_load_11[3];
                    float _vec_load_12[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (ldst + 16) + 0);
                        _vec_load_12[0 + 0] = _v4.x;
                        _vec_load_12[0 + 1] = _v4.y;
                        _vec_load_12[0 + 2] = _v4.z;
                        _vec_load_12[0 + 3] = _v4.w;
                    }
                    nat[16] = nat[16] + _vec_load_12[0];
                    nat[17] = nat[17] + _vec_load_12[1];
                    nat[18] = nat[18] + _vec_load_12[2];
                    nat[19] = nat[19] + _vec_load_12[3];
                    float _vec_load_13[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (ldst + 20) + 0);
                        _vec_load_13[0 + 0] = _v4.x;
                        _vec_load_13[0 + 1] = _v4.y;
                        _vec_load_13[0 + 2] = _v4.z;
                        _vec_load_13[0 + 3] = _v4.w;
                    }
                    nat[20] = nat[20] + _vec_load_13[0];
                    nat[21] = nat[21] + _vec_load_13[1];
                    nat[22] = nat[22] + _vec_load_13[2];
                    nat[23] = nat[23] + _vec_load_13[3];
                    float _vec_load_14[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (ldst + 24) + 0);
                        _vec_load_14[0 + 0] = _v4.x;
                        _vec_load_14[0 + 1] = _v4.y;
                        _vec_load_14[0 + 2] = _v4.z;
                        _vec_load_14[0 + 3] = _v4.w;
                    }
                    nat[24] = nat[24] + _vec_load_14[0];
                    nat[25] = nat[25] + _vec_load_14[1];
                    nat[26] = nat[26] + _vec_load_14[2];
                    nat[27] = nat[27] + _vec_load_14[3];
                    float _vec_load_15[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (ldst + 28) + 0);
                        _vec_load_15[0 + 0] = _v4.x;
                        _vec_load_15[0 + 1] = _v4.y;
                        _vec_load_15[0 + 2] = _v4.z;
                        _vec_load_15[0 + 3] = _v4.w;
                    }
                    nat[28] = nat[28] + _vec_load_15[0];
                    nat[29] = nat[29] + _vec_load_15[1];
                    nat[30] = nat[30] + _vec_load_15[2];
                    nat[31] = nat[31] + _vec_load_15[3];
                    {
                        float4 _v4 = make_float4(nat[0 + 0], nat[0 + 1], nat[0 + 2], nat[0 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + ldst) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(nat[4 + 0], nat[4 + 1], nat[4 + 2], nat[4 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + ldst + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(nat[8 + 0], nat[8 + 1], nat[8 + 2], nat[8 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + ldst + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(nat[12 + 0], nat[12 + 1], nat[12 + 2], nat[12 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + ldst + 12) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(nat[16 + 0], nat[16 + 1], nat[16 + 2], nat[16 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + ldst + 16) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(nat[20 + 0], nat[20 + 1], nat[20 + 2], nat[20 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + ldst + 20) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(nat[24 + 0], nat[24 + 1], nat[24 + 2], nat[24 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + ldst + 24) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(nat[28 + 0], nat[28 + 1], nat[28 + 2], nat[28 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + ldst + 28) = _v4;
                    }
                }
            }
        } else if (out_f32 != 0) {
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
        float _vec_load_16[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + rbase + 0);
            _vec_load_16[0 + 0] = _v4.x;
            _vec_load_16[0 + 1] = _v4.y;
            _vec_load_16[0 + 2] = _v4.z;
            _vec_load_16[0 + 3] = _v4.w;
        }
        rvals[0] = _vec_load_16[0];
        rvals[1] = _vec_load_16[1];
        rvals[2] = _vec_load_16[2];
        rvals[3] = _vec_load_16[3];
        float _vec_load_17[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 4) + 0);
            _vec_load_17[0 + 0] = _v4.x;
            _vec_load_17[0 + 1] = _v4.y;
            _vec_load_17[0 + 2] = _v4.z;
            _vec_load_17[0 + 3] = _v4.w;
        }
        rvals[4] = _vec_load_17[0];
        rvals[5] = _vec_load_17[1];
        rvals[6] = _vec_load_17[2];
        rvals[7] = _vec_load_17[3];
        float _vec_load_18[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 8) + 0);
            _vec_load_18[0 + 0] = _v4.x;
            _vec_load_18[0 + 1] = _v4.y;
            _vec_load_18[0 + 2] = _v4.z;
            _vec_load_18[0 + 3] = _v4.w;
        }
        rvals[8] = _vec_load_18[0];
        rvals[9] = _vec_load_18[1];
        rvals[10] = _vec_load_18[2];
        rvals[11] = _vec_load_18[3];
        float _vec_load_19[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 12) + 0);
            _vec_load_19[0 + 0] = _v4.x;
            _vec_load_19[0 + 1] = _v4.y;
            _vec_load_19[0 + 2] = _v4.z;
            _vec_load_19[0 + 3] = _v4.w;
        }
        rvals[12] = _vec_load_19[0];
        rvals[13] = _vec_load_19[1];
        rvals[14] = _vec_load_19[2];
        rvals[15] = _vec_load_19[3];
        float _vec_load_20[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 16) + 0);
            _vec_load_20[0 + 0] = _v4.x;
            _vec_load_20[0 + 1] = _v4.y;
            _vec_load_20[0 + 2] = _v4.z;
            _vec_load_20[0 + 3] = _v4.w;
        }
        rvals[16] = _vec_load_20[0];
        rvals[17] = _vec_load_20[1];
        rvals[18] = _vec_load_20[2];
        rvals[19] = _vec_load_20[3];
        float _vec_load_21[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 20) + 0);
            _vec_load_21[0 + 0] = _v4.x;
            _vec_load_21[0 + 1] = _v4.y;
            _vec_load_21[0 + 2] = _v4.z;
            _vec_load_21[0 + 3] = _v4.w;
        }
        rvals[20] = _vec_load_21[0];
        rvals[21] = _vec_load_21[1];
        rvals[22] = _vec_load_21[2];
        rvals[23] = _vec_load_21[3];
        float _vec_load_22[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 24) + 0);
            _vec_load_22[0 + 0] = _v4.x;
            _vec_load_22[0 + 1] = _v4.y;
            _vec_load_22[0 + 2] = _v4.z;
            _vec_load_22[0 + 3] = _v4.w;
        }
        rvals[24] = _vec_load_22[0];
        rvals[25] = _vec_load_22[1];
        rvals[26] = _vec_load_22[2];
        rvals[27] = _vec_load_22[3];
        float _vec_load_23[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(src_rope + (rbase + 28) + 0);
            _vec_load_23[0 + 0] = _v4.x;
            _vec_load_23[0 + 1] = _v4.y;
            _vec_load_23[0 + 2] = _v4.z;
            _vec_load_23[0 + 3] = _v4.w;
        }
        rvals[28] = _vec_load_23[0];
        rvals[29] = _vec_load_23[1];
        rvals[30] = _vec_load_23[2];
        rvals[31] = _vec_load_23[3];
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
        if (accumulate != 0) {
            float rnz = 1.0f;
            {
                float _fabs_32 = fabsf(rvals[0]);
                rnz = _fabs_32;
                float _fabs_33 = fabsf(rvals[1]);
                rnz = rnz + _fabs_33;
                float _fabs_34 = fabsf(rvals[2]);
                rnz = rnz + _fabs_34;
                float _fabs_35 = fabsf(rvals[3]);
                rnz = rnz + _fabs_35;
                float _fabs_36 = fabsf(rvals[4]);
                rnz = rnz + _fabs_36;
                float _fabs_37 = fabsf(rvals[5]);
                rnz = rnz + _fabs_37;
                float _fabs_38 = fabsf(rvals[6]);
                rnz = rnz + _fabs_38;
                float _fabs_39 = fabsf(rvals[7]);
                rnz = rnz + _fabs_39;
                float _fabs_40 = fabsf(rvals[8]);
                rnz = rnz + _fabs_40;
                float _fabs_41 = fabsf(rvals[9]);
                rnz = rnz + _fabs_41;
                float _fabs_42 = fabsf(rvals[10]);
                rnz = rnz + _fabs_42;
                float _fabs_43 = fabsf(rvals[11]);
                rnz = rnz + _fabs_43;
                float _fabs_44 = fabsf(rvals[12]);
                rnz = rnz + _fabs_44;
                float _fabs_45 = fabsf(rvals[13]);
                rnz = rnz + _fabs_45;
                float _fabs_46 = fabsf(rvals[14]);
                rnz = rnz + _fabs_46;
                float _fabs_47 = fabsf(rvals[15]);
                rnz = rnz + _fabs_47;
                float _fabs_48 = fabsf(rvals[16]);
                rnz = rnz + _fabs_48;
                float _fabs_49 = fabsf(rvals[17]);
                rnz = rnz + _fabs_49;
                float _fabs_50 = fabsf(rvals[18]);
                rnz = rnz + _fabs_50;
                float _fabs_51 = fabsf(rvals[19]);
                rnz = rnz + _fabs_51;
                float _fabs_52 = fabsf(rvals[20]);
                rnz = rnz + _fabs_52;
                float _fabs_53 = fabsf(rvals[21]);
                rnz = rnz + _fabs_53;
                float _fabs_54 = fabsf(rvals[22]);
                rnz = rnz + _fabs_54;
                float _fabs_55 = fabsf(rvals[23]);
                rnz = rnz + _fabs_55;
                float _fabs_56 = fabsf(rvals[24]);
                rnz = rnz + _fabs_56;
                float _fabs_57 = fabsf(rvals[25]);
                rnz = rnz + _fabs_57;
                float _fabs_58 = fabsf(rvals[26]);
                rnz = rnz + _fabs_58;
                float _fabs_59 = fabsf(rvals[27]);
                rnz = rnz + _fabs_59;
                float _fabs_60 = fabsf(rvals[28]);
                rnz = rnz + _fabs_60;
                float _fabs_61 = fabsf(rvals[29]);
                rnz = rnz + _fabs_61;
                float _fabs_62 = fabsf(rvals[30]);
                rnz = rnz + _fabs_62;
                float _fabs_63 = fabsf(rvals[31]);
                rnz = rnz + _fabs_63;
            }
            if (rnz != 0.0f) {
                int rg = g - latent_groups;
                int rrow = rg / 2;
                if (has_dst_map != 0) {
                    rrow = dst_map[(long long)rrow];
                }
                long long rdst = (long long)rrow * (long long)dst_row_stride + (long long)(rg % 2) * 32 + 512;
                if (has_dst_map != 0) {
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[rdst])), "f"(rnat[0]), "f"(rnat[1]), "f"(rnat[2]), "f"(rnat[3]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[rdst + 4])), "f"(rnat[4]), "f"(rnat[5]), "f"(rnat[6]), "f"(rnat[7]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[rdst + 8])), "f"(rnat[8]), "f"(rnat[9]), "f"(rnat[10]), "f"(rnat[11]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[rdst + 12])), "f"(rnat[12]), "f"(rnat[13]), "f"(rnat[14]), "f"(rnat[15]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[rdst + 16])), "f"(rnat[16]), "f"(rnat[17]), "f"(rnat[18]), "f"(rnat[19]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[rdst + 20])), "f"(rnat[20]), "f"(rnat[21]), "f"(rnat[22]), "f"(rnat[23]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[rdst + 24])), "f"(rnat[24]), "f"(rnat[25]), "f"(rnat[26]), "f"(rnat[27]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dst_packed[rdst + 28])), "f"(rnat[28]), "f"(rnat[29]), "f"(rnat[30]), "f"(rnat[31]) : "memory");
                } else {
                    float _vec_load_24[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + rdst + 0);
                        _vec_load_24[0 + 0] = _v4.x;
                        _vec_load_24[0 + 1] = _v4.y;
                        _vec_load_24[0 + 2] = _v4.z;
                        _vec_load_24[0 + 3] = _v4.w;
                    }
                    rnat[0] = rnat[0] + _vec_load_24[0];
                    rnat[1] = rnat[1] + _vec_load_24[1];
                    rnat[2] = rnat[2] + _vec_load_24[2];
                    rnat[3] = rnat[3] + _vec_load_24[3];
                    float _vec_load_25[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (rdst + 4) + 0);
                        _vec_load_25[0 + 0] = _v4.x;
                        _vec_load_25[0 + 1] = _v4.y;
                        _vec_load_25[0 + 2] = _v4.z;
                        _vec_load_25[0 + 3] = _v4.w;
                    }
                    rnat[4] = rnat[4] + _vec_load_25[0];
                    rnat[5] = rnat[5] + _vec_load_25[1];
                    rnat[6] = rnat[6] + _vec_load_25[2];
                    rnat[7] = rnat[7] + _vec_load_25[3];
                    float _vec_load_26[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (rdst + 8) + 0);
                        _vec_load_26[0 + 0] = _v4.x;
                        _vec_load_26[0 + 1] = _v4.y;
                        _vec_load_26[0 + 2] = _v4.z;
                        _vec_load_26[0 + 3] = _v4.w;
                    }
                    rnat[8] = rnat[8] + _vec_load_26[0];
                    rnat[9] = rnat[9] + _vec_load_26[1];
                    rnat[10] = rnat[10] + _vec_load_26[2];
                    rnat[11] = rnat[11] + _vec_load_26[3];
                    float _vec_load_27[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (rdst + 12) + 0);
                        _vec_load_27[0 + 0] = _v4.x;
                        _vec_load_27[0 + 1] = _v4.y;
                        _vec_load_27[0 + 2] = _v4.z;
                        _vec_load_27[0 + 3] = _v4.w;
                    }
                    rnat[12] = rnat[12] + _vec_load_27[0];
                    rnat[13] = rnat[13] + _vec_load_27[1];
                    rnat[14] = rnat[14] + _vec_load_27[2];
                    rnat[15] = rnat[15] + _vec_load_27[3];
                    float _vec_load_28[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (rdst + 16) + 0);
                        _vec_load_28[0 + 0] = _v4.x;
                        _vec_load_28[0 + 1] = _v4.y;
                        _vec_load_28[0 + 2] = _v4.z;
                        _vec_load_28[0 + 3] = _v4.w;
                    }
                    rnat[16] = rnat[16] + _vec_load_28[0];
                    rnat[17] = rnat[17] + _vec_load_28[1];
                    rnat[18] = rnat[18] + _vec_load_28[2];
                    rnat[19] = rnat[19] + _vec_load_28[3];
                    float _vec_load_29[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (rdst + 20) + 0);
                        _vec_load_29[0 + 0] = _v4.x;
                        _vec_load_29[0 + 1] = _v4.y;
                        _vec_load_29[0 + 2] = _v4.z;
                        _vec_load_29[0 + 3] = _v4.w;
                    }
                    rnat[20] = rnat[20] + _vec_load_29[0];
                    rnat[21] = rnat[21] + _vec_load_29[1];
                    rnat[22] = rnat[22] + _vec_load_29[2];
                    rnat[23] = rnat[23] + _vec_load_29[3];
                    float _vec_load_30[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (rdst + 24) + 0);
                        _vec_load_30[0 + 0] = _v4.x;
                        _vec_load_30[0 + 1] = _v4.y;
                        _vec_load_30[0 + 2] = _v4.z;
                        _vec_load_30[0 + 3] = _v4.w;
                    }
                    rnat[24] = rnat[24] + _vec_load_30[0];
                    rnat[25] = rnat[25] + _vec_load_30[1];
                    rnat[26] = rnat[26] + _vec_load_30[2];
                    rnat[27] = rnat[27] + _vec_load_30[3];
                    float _vec_load_31[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + (rdst + 28) + 0);
                        _vec_load_31[0 + 0] = _v4.x;
                        _vec_load_31[0 + 1] = _v4.y;
                        _vec_load_31[0 + 2] = _v4.z;
                        _vec_load_31[0 + 3] = _v4.w;
                    }
                    rnat[28] = rnat[28] + _vec_load_31[0];
                    rnat[29] = rnat[29] + _vec_load_31[1];
                    rnat[30] = rnat[30] + _vec_load_31[2];
                    rnat[31] = rnat[31] + _vec_load_31[3];
                    {
                        float4 _v4 = make_float4(rnat[0 + 0], rnat[0 + 1], rnat[0 + 2], rnat[0 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + rdst) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(rnat[4 + 0], rnat[4 + 1], rnat[4 + 2], rnat[4 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + rdst + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(rnat[8 + 0], rnat[8 + 1], rnat[8 + 2], rnat[8 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + rdst + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(rnat[12 + 0], rnat[12 + 1], rnat[12 + 2], rnat[12 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + rdst + 12) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(rnat[16 + 0], rnat[16 + 1], rnat[16 + 2], rnat[16 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + rdst + 16) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(rnat[20 + 0], rnat[20 + 1], rnat[20 + 2], rnat[20 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + rdst + 20) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(rnat[24 + 0], rnat[24 + 1], rnat[24 + 2], rnat[24 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + rdst + 24) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(rnat[28 + 0], rnat[28 + 1], rnat[28 + 2], rnat[28 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + rdst + 28) = _v4;
                    }
                }
            }
        } else if (out_f32 != 0) {
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
