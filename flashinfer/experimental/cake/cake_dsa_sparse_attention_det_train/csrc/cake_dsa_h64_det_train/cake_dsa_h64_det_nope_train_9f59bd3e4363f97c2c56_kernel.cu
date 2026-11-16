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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_det_nope_train_9f59bd3e4363f97c2c56(int* dkv_acc, int* __restrict__ e_lat, unsigned long long* __restrict__ cnt64, unsigned int* __restrict__ row_pairs, __nv_bfloat16* dst_latent, float* dst_latent_f32, float* dst_packed, int num_rows, int dkv_stride, int dst_lat_stride, int out_f32, int dst_row_stride, int accumulate, int* __restrict__ redo_rows, int* __restrict__ redo_mark, unsigned int* __restrict__ det_globals, int capacity)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int first = blockIdx.x * 8 + warp;
    int stride = gridDim.x * 8;
    int count = (int)det_globals[8];
    if (count <= capacity) {
        #pragma unroll 1
        for (int i = first; i < count; i += stride) {
            int list_row = redo_rows[i];
            int lane_1 = lane;
            int word0 = e_lat[list_row];
            int e = word0 << 16 >> 16;
            int touched = (int)(e != 127);
            int n[16];
            float prev[16];
            unsigned long long worst = 0ull;
            unsigned int pairs = 0;
            int e_r = 0;
            if (touched != 0) {
                unsigned long long cq_0 = cnt64[(long long)list_row * 4ll];
                unsigned long long _max_0 = ((worst) > (cq_0) ? (worst) : (cq_0));
                worst = _max_0;
                unsigned long long cq_1 = cnt64[(long long)list_row * 4ll + 1ll];
                unsigned long long _max_1 = ((worst) > (cq_1) ? (worst) : (cq_1));
                worst = _max_1;
                unsigned long long cq_2 = cnt64[(long long)list_row * 4ll + 2ll];
                unsigned long long _max_2 = ((worst) > (cq_2) ? (worst) : (cq_2));
                worst = _max_2;
                unsigned long long cq_3 = cnt64[(long long)list_row * 4ll + 3ll];
                unsigned long long _max_3 = ((worst) > (cq_3) ? (worst) : (cq_3));
                worst = _max_3;
                pairs = row_pairs[list_row];
                {
                    const int4* _ivptr_4 = reinterpret_cast<const int4*>(dkv_acc + ((long long)list_row * (long long)dkv_stride + (long long)(lane_1 * 4)) + 0);
                    int4 _ivld_4;
                    _ivld_4 = *_ivptr_4;
                    n[0 + 0] = _ivld_4.x;
                    n[0 + 1] = _ivld_4.y;
                    n[0 + 2] = _ivld_4.z;
                    n[0 + 3] = _ivld_4.w;
                }
                {
                    const int4* _ivptr_5 = reinterpret_cast<const int4*>(dkv_acc + ((long long)list_row * (long long)dkv_stride + (long long)(128 + lane_1 * 4)) + 0);
                    int4 _ivld_5;
                    _ivld_5 = *_ivptr_5;
                    n[4 + 0] = _ivld_5.x;
                    n[4 + 1] = _ivld_5.y;
                    n[4 + 2] = _ivld_5.z;
                    n[4 + 3] = _ivld_5.w;
                }
                {
                    const int4* _ivptr_6 = reinterpret_cast<const int4*>(dkv_acc + ((long long)list_row * (long long)dkv_stride + (long long)(256 + lane_1 * 4)) + 0);
                    int4 _ivld_6;
                    _ivld_6 = *_ivptr_6;
                    n[8 + 0] = _ivld_6.x;
                    n[8 + 1] = _ivld_6.y;
                    n[8 + 2] = _ivld_6.z;
                    n[8 + 3] = _ivld_6.w;
                }
                {
                    const int4* _ivptr_7 = reinterpret_cast<const int4*>(dkv_acc + ((long long)list_row * (long long)dkv_stride + (long long)(384 + lane_1 * 4)) + 0);
                    int4 _ivld_7;
                    _ivld_7 = *_ivptr_7;
                    n[12 + 0] = _ivld_7.x;
                    n[12 + 1] = _ivld_7.y;
                    n[12 + 2] = _ivld_7.z;
                    n[12 + 3] = _ivld_7.w;
                }
                if (accumulate != 0) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)list_row * (long long)dst_row_stride + (long long)(lane_1 * 4)) + 0);
                        prev[0 + 0] = _v4.x;
                        prev[0 + 1] = _v4.y;
                        prev[0 + 2] = _v4.z;
                        prev[0 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)list_row * (long long)dst_row_stride + (long long)(128 + lane_1 * 4)) + 0);
                        prev[4 + 0] = _v4.x;
                        prev[4 + 1] = _v4.y;
                        prev[4 + 2] = _v4.z;
                        prev[4 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)list_row * (long long)dst_row_stride + (long long)(256 + lane_1 * 4)) + 0);
                        prev[8 + 0] = _v4.x;
                        prev[8 + 1] = _v4.y;
                        prev[8 + 2] = _v4.z;
                        prev[8 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)list_row * (long long)dst_row_stride + (long long)(384 + lane_1 * 4)) + 0);
                        prev[12 + 0] = _v4.x;
                        prev[12 + 1] = _v4.y;
                        prev[12 + 2] = _v4.z;
                        prev[12 + 3] = _v4.w;
                    }
                }
            } else {
                n[0] = 0;
                n[1] = 0;
                n[2] = 0;
                n[3] = 0;
                n[4] = 0;
                n[5] = 0;
                n[6] = 0;
                n[7] = 0;
                n[8] = 0;
                n[9] = 0;
                n[10] = 0;
                n[11] = 0;
                n[12] = 0;
                n[13] = 0;
                n[14] = 0;
                n[15] = 0;
            }
            int bad = 0;
            if (touched != 0) {
                unsigned long long slack = (unsigned long long)pairs << (unsigned long long)(word0 >> 16);
                if (worst >= 2147483648ull || worst + slack >= 2147483648ull) {
                    bad = 1;
                }
            }
            float nan = __uint_as_float(2143289344);
            float s_lat = 0.0f;
            if (touched != 0) {
                s_lat = __uint_as_float((unsigned int)(e + 127) << 23);
            }
            int col_0 = lane_1 * 4;
            float v4[4];
            v4[0] = mul_f32_rn_noftz((float)n[0], s_lat);
            if (bad != 0) {
                v4[0] = nan;
            }
            v4[1] = mul_f32_rn_noftz((float)n[1], s_lat);
            if (bad != 0) {
                v4[1] = nan;
            }
            v4[2] = mul_f32_rn_noftz((float)n[2], s_lat);
            if (bad != 0) {
                v4[2] = nan;
            }
            v4[3] = mul_f32_rn_noftz((float)n[3], s_lat);
            if (bad != 0) {
                v4[3] = nan;
            }
            if (accumulate != 0) {
                if (touched != 0) {
                    if (n[0] != 0 || bad != 0) {
                        float _add_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(prev[0]), "f"(v4[0]));
                        prev[0] = _add_12;
                    }
                    if (n[1] != 0 || bad != 0) {
                        float _add_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(prev[1]), "f"(v4[1]));
                        prev[1] = _add_13;
                    }
                    if (n[2] != 0 || bad != 0) {
                        float _add_14;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(prev[2]), "f"(v4[2]));
                        prev[2] = _add_14;
                    }
                    if (n[3] != 0 || bad != 0) {
                        float _add_15;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(prev[3]), "f"(v4[3]));
                        prev[3] = _add_15;
                    }
                    {
                        float4 _v4 = make_float4(prev[0 + 0], prev[0 + 1], prev[0 + 2], prev[0 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + (long long)list_row * (long long)dst_row_stride + (long long)col_0) = _v4;
                    }
                }
            } else if (out_f32 != 0) {
                {
                    float4 _v4 = make_float4(v4[0 + 0], v4[0 + 1], v4[0 + 2], v4[0 + 3]);
                    *reinterpret_cast<float4*>(dst_latent_f32 + (long long)list_row * (long long)dst_lat_stride + (long long)col_0) = _v4;
                }
            } else {
                unsigned int packed[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4[_lp*2 + 0], v4[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dst_latent + ((long long)list_row * (long long)dst_lat_stride + (long long)col_0))[0] = reinterpret_cast<int2*>(packed)[0];
            }
            int col_1 = 128 + lane_1 * 4;
            float v4_0[4];
            v4_0[0] = mul_f32_rn_noftz((float)n[4], s_lat);
            if (bad != 0) {
                v4_0[0] = nan;
            }
            v4_0[1] = mul_f32_rn_noftz((float)n[5], s_lat);
            if (bad != 0) {
                v4_0[1] = nan;
            }
            v4_0[2] = mul_f32_rn_noftz((float)n[6], s_lat);
            if (bad != 0) {
                v4_0[2] = nan;
            }
            v4_0[3] = mul_f32_rn_noftz((float)n[7], s_lat);
            if (bad != 0) {
                v4_0[3] = nan;
            }
            if (accumulate != 0) {
                if (touched != 0) {
                    if (n[4] != 0 || bad != 0) {
                        float _add_16;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(prev[4]), "f"(v4_0[0]));
                        prev[4] = _add_16;
                    }
                    if (n[5] != 0 || bad != 0) {
                        float _add_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(prev[5]), "f"(v4_0[1]));
                        prev[5] = _add_17;
                    }
                    if (n[6] != 0 || bad != 0) {
                        float _add_18;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(prev[6]), "f"(v4_0[2]));
                        prev[6] = _add_18;
                    }
                    if (n[7] != 0 || bad != 0) {
                        float _add_19;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(prev[7]), "f"(v4_0[3]));
                        prev[7] = _add_19;
                    }
                    {
                        float4 _v4 = make_float4(prev[4 + 0], prev[4 + 1], prev[4 + 2], prev[4 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + (long long)list_row * (long long)dst_row_stride + (long long)col_1) = _v4;
                    }
                }
            } else if (out_f32 != 0) {
                {
                    float4 _v4 = make_float4(v4_0[0 + 0], v4_0[0 + 1], v4_0[0 + 2], v4_0[0 + 3]);
                    *reinterpret_cast<float4*>(dst_latent_f32 + (long long)list_row * (long long)dst_lat_stride + (long long)col_1) = _v4;
                }
            } else {
                unsigned int packed_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_0[_lp*2 + 0], v4_0[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dst_latent + ((long long)list_row * (long long)dst_lat_stride + (long long)col_1))[0] = reinterpret_cast<int2*>(packed_1)[0];
            }
            int col_2 = 256 + lane_1 * 4;
            float v4_1[4];
            v4_1[0] = mul_f32_rn_noftz((float)n[8], s_lat);
            if (bad != 0) {
                v4_1[0] = nan;
            }
            v4_1[1] = mul_f32_rn_noftz((float)n[9], s_lat);
            if (bad != 0) {
                v4_1[1] = nan;
            }
            v4_1[2] = mul_f32_rn_noftz((float)n[10], s_lat);
            if (bad != 0) {
                v4_1[2] = nan;
            }
            v4_1[3] = mul_f32_rn_noftz((float)n[11], s_lat);
            if (bad != 0) {
                v4_1[3] = nan;
            }
            if (accumulate != 0) {
                if (touched != 0) {
                    if (n[8] != 0 || bad != 0) {
                        float _add_20;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(prev[8]), "f"(v4_1[0]));
                        prev[8] = _add_20;
                    }
                    if (n[9] != 0 || bad != 0) {
                        float _add_21;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(prev[9]), "f"(v4_1[1]));
                        prev[9] = _add_21;
                    }
                    if (n[10] != 0 || bad != 0) {
                        float _add_22;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(prev[10]), "f"(v4_1[2]));
                        prev[10] = _add_22;
                    }
                    if (n[11] != 0 || bad != 0) {
                        float _add_23;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(prev[11]), "f"(v4_1[3]));
                        prev[11] = _add_23;
                    }
                    {
                        float4 _v4 = make_float4(prev[8 + 0], prev[8 + 1], prev[8 + 2], prev[8 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + (long long)list_row * (long long)dst_row_stride + (long long)col_2) = _v4;
                    }
                }
            } else if (out_f32 != 0) {
                {
                    float4 _v4 = make_float4(v4_1[0 + 0], v4_1[0 + 1], v4_1[0 + 2], v4_1[0 + 3]);
                    *reinterpret_cast<float4*>(dst_latent_f32 + (long long)list_row * (long long)dst_lat_stride + (long long)col_2) = _v4;
                }
            } else {
                unsigned int packed_2[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_1[_lp*2 + 0], v4_1[_lp*2+1 + 0]));
                    packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dst_latent + ((long long)list_row * (long long)dst_lat_stride + (long long)col_2))[0] = reinterpret_cast<int2*>(packed_2)[0];
            }
            int col_3 = 384 + lane_1 * 4;
            float v4_2[4];
            v4_2[0] = mul_f32_rn_noftz((float)n[12], s_lat);
            if (bad != 0) {
                v4_2[0] = nan;
            }
            v4_2[1] = mul_f32_rn_noftz((float)n[13], s_lat);
            if (bad != 0) {
                v4_2[1] = nan;
            }
            v4_2[2] = mul_f32_rn_noftz((float)n[14], s_lat);
            if (bad != 0) {
                v4_2[2] = nan;
            }
            v4_2[3] = mul_f32_rn_noftz((float)n[15], s_lat);
            if (bad != 0) {
                v4_2[3] = nan;
            }
            if (accumulate != 0) {
                if (touched != 0) {
                    if (n[12] != 0 || bad != 0) {
                        float _add_24;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(prev[12]), "f"(v4_2[0]));
                        prev[12] = _add_24;
                    }
                    if (n[13] != 0 || bad != 0) {
                        float _add_25;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(prev[13]), "f"(v4_2[1]));
                        prev[13] = _add_25;
                    }
                    if (n[14] != 0 || bad != 0) {
                        float _add_26;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(prev[14]), "f"(v4_2[2]));
                        prev[14] = _add_26;
                    }
                    if (n[15] != 0 || bad != 0) {
                        float _add_27;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(prev[15]), "f"(v4_2[3]));
                        prev[15] = _add_27;
                    }
                    {
                        float4 _v4 = make_float4(prev[12 + 0], prev[12 + 1], prev[12 + 2], prev[12 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + (long long)list_row * (long long)dst_row_stride + (long long)col_3) = _v4;
                    }
                }
            } else if (out_f32 != 0) {
                {
                    float4 _v4 = make_float4(v4_2[0 + 0], v4_2[0 + 1], v4_2[0 + 2], v4_2[0 + 3]);
                    *reinterpret_cast<float4*>(dst_latent_f32 + (long long)list_row * (long long)dst_lat_stride + (long long)col_3) = _v4;
                }
            } else {
                unsigned int packed_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_2[_lp*2 + 0], v4_2[_lp*2+1 + 0]));
                    packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dst_latent + ((long long)list_row * (long long)dst_lat_stride + (long long)col_3))[0] = reinterpret_cast<int2*>(packed_3)[0];
            }
        }
    } else {
        #pragma unroll 1
        for (int r = first; r < num_rows; r += stride) {
            if (redo_mark[r] != 0) {
                int lane_2 = lane;
                int word0_1 = e_lat[r];
                int e_1 = word0_1 << 16 >> 16;
                int touched_1 = (int)(e_1 != 127);
                int n_1[16];
                float prev_1[16];
                unsigned long long worst_1 = 0ull;
                unsigned int pairs_1 = 0;
                int e_r_1 = 0;
                if (touched_1 != 0) {
                    unsigned long long cq_4 = cnt64[(long long)r * 4ll];
                    unsigned long long _max_28 = ((worst_1) > (cq_4) ? (worst_1) : (cq_4));
                    worst_1 = _max_28;
                    unsigned long long cq_5 = cnt64[(long long)r * 4ll + 1ll];
                    unsigned long long _max_29 = ((worst_1) > (cq_5) ? (worst_1) : (cq_5));
                    worst_1 = _max_29;
                    unsigned long long cq_6 = cnt64[(long long)r * 4ll + 2ll];
                    unsigned long long _max_30 = ((worst_1) > (cq_6) ? (worst_1) : (cq_6));
                    worst_1 = _max_30;
                    unsigned long long cq_7 = cnt64[(long long)r * 4ll + 3ll];
                    unsigned long long _max_31 = ((worst_1) > (cq_7) ? (worst_1) : (cq_7));
                    worst_1 = _max_31;
                    pairs_1 = row_pairs[r];
                    {
                        const int4* _ivptr_32 = reinterpret_cast<const int4*>(dkv_acc + ((long long)r * (long long)dkv_stride + (long long)(lane_2 * 4)) + 0);
                        int4 _ivld_32;
                        _ivld_32 = *_ivptr_32;
                        n_1[0 + 0] = _ivld_32.x;
                        n_1[0 + 1] = _ivld_32.y;
                        n_1[0 + 2] = _ivld_32.z;
                        n_1[0 + 3] = _ivld_32.w;
                    }
                    {
                        const int4* _ivptr_33 = reinterpret_cast<const int4*>(dkv_acc + ((long long)r * (long long)dkv_stride + (long long)(128 + lane_2 * 4)) + 0);
                        int4 _ivld_33;
                        _ivld_33 = *_ivptr_33;
                        n_1[4 + 0] = _ivld_33.x;
                        n_1[4 + 1] = _ivld_33.y;
                        n_1[4 + 2] = _ivld_33.z;
                        n_1[4 + 3] = _ivld_33.w;
                    }
                    {
                        const int4* _ivptr_34 = reinterpret_cast<const int4*>(dkv_acc + ((long long)r * (long long)dkv_stride + (long long)(256 + lane_2 * 4)) + 0);
                        int4 _ivld_34;
                        _ivld_34 = *_ivptr_34;
                        n_1[8 + 0] = _ivld_34.x;
                        n_1[8 + 1] = _ivld_34.y;
                        n_1[8 + 2] = _ivld_34.z;
                        n_1[8 + 3] = _ivld_34.w;
                    }
                    {
                        const int4* _ivptr_35 = reinterpret_cast<const int4*>(dkv_acc + ((long long)r * (long long)dkv_stride + (long long)(384 + lane_2 * 4)) + 0);
                        int4 _ivld_35;
                        _ivld_35 = *_ivptr_35;
                        n_1[12 + 0] = _ivld_35.x;
                        n_1[12 + 1] = _ivld_35.y;
                        n_1[12 + 2] = _ivld_35.z;
                        n_1[12 + 3] = _ivld_35.w;
                    }
                    if (accumulate != 0) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)r * (long long)dst_row_stride + (long long)(lane_2 * 4)) + 0);
                            prev_1[0 + 0] = _v4.x;
                            prev_1[0 + 1] = _v4.y;
                            prev_1[0 + 2] = _v4.z;
                            prev_1[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)r * (long long)dst_row_stride + (long long)(128 + lane_2 * 4)) + 0);
                            prev_1[4 + 0] = _v4.x;
                            prev_1[4 + 1] = _v4.y;
                            prev_1[4 + 2] = _v4.z;
                            prev_1[4 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)r * (long long)dst_row_stride + (long long)(256 + lane_2 * 4)) + 0);
                            prev_1[8 + 0] = _v4.x;
                            prev_1[8 + 1] = _v4.y;
                            prev_1[8 + 2] = _v4.z;
                            prev_1[8 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)r * (long long)dst_row_stride + (long long)(384 + lane_2 * 4)) + 0);
                            prev_1[12 + 0] = _v4.x;
                            prev_1[12 + 1] = _v4.y;
                            prev_1[12 + 2] = _v4.z;
                            prev_1[12 + 3] = _v4.w;
                        }
                    }
                } else {
                    n_1[0] = 0;
                    n_1[1] = 0;
                    n_1[2] = 0;
                    n_1[3] = 0;
                    n_1[4] = 0;
                    n_1[5] = 0;
                    n_1[6] = 0;
                    n_1[7] = 0;
                    n_1[8] = 0;
                    n_1[9] = 0;
                    n_1[10] = 0;
                    n_1[11] = 0;
                    n_1[12] = 0;
                    n_1[13] = 0;
                    n_1[14] = 0;
                    n_1[15] = 0;
                }
                int bad_1 = 0;
                if (touched_1 != 0) {
                    unsigned long long slack_1 = (unsigned long long)pairs_1 << (unsigned long long)(word0_1 >> 16);
                    if (worst_1 >= 2147483648ull || worst_1 + slack_1 >= 2147483648ull) {
                        bad_1 = 1;
                    }
                }
                float nan_1 = __uint_as_float(2143289344);
                float s_lat_1 = 0.0f;
                if (touched_1 != 0) {
                    s_lat_1 = __uint_as_float((unsigned int)(e_1 + 127) << 23);
                }
                int col_4 = lane_2 * 4;
                float v4_3[4];
                v4_3[0] = mul_f32_rn_noftz((float)n_1[0], s_lat_1);
                if (bad_1 != 0) {
                    v4_3[0] = nan_1;
                }
                v4_3[1] = mul_f32_rn_noftz((float)n_1[1], s_lat_1);
                if (bad_1 != 0) {
                    v4_3[1] = nan_1;
                }
                v4_3[2] = mul_f32_rn_noftz((float)n_1[2], s_lat_1);
                if (bad_1 != 0) {
                    v4_3[2] = nan_1;
                }
                v4_3[3] = mul_f32_rn_noftz((float)n_1[3], s_lat_1);
                if (bad_1 != 0) {
                    v4_3[3] = nan_1;
                }
                if (accumulate != 0) {
                    if (touched_1 != 0) {
                        if (n_1[0] != 0 || bad_1 != 0) {
                            float _add_40;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(prev_1[0]), "f"(v4_3[0]));
                            prev_1[0] = _add_40;
                        }
                        if (n_1[1] != 0 || bad_1 != 0) {
                            float _add_41;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(prev_1[1]), "f"(v4_3[1]));
                            prev_1[1] = _add_41;
                        }
                        if (n_1[2] != 0 || bad_1 != 0) {
                            float _add_42;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(prev_1[2]), "f"(v4_3[2]));
                            prev_1[2] = _add_42;
                        }
                        if (n_1[3] != 0 || bad_1 != 0) {
                            float _add_43;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(prev_1[3]), "f"(v4_3[3]));
                            prev_1[3] = _add_43;
                        }
                        {
                            float4 _v4 = make_float4(prev_1[0 + 0], prev_1[0 + 1], prev_1[0 + 2], prev_1[0 + 3]);
                            *reinterpret_cast<float4*>(dst_packed + (long long)r * (long long)dst_row_stride + (long long)col_4) = _v4;
                        }
                    }
                } else if (out_f32 != 0) {
                    {
                        float4 _v4 = make_float4(v4_3[0 + 0], v4_3[0 + 1], v4_3[0 + 2], v4_3[0 + 3]);
                        *reinterpret_cast<float4*>(dst_latent_f32 + (long long)r * (long long)dst_lat_stride + (long long)col_4) = _v4;
                    }
                } else {
                    unsigned int packed_4[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_3[_lp*2 + 0], v4_3[_lp*2+1 + 0]));
                        packed_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int2*>(dst_latent + ((long long)r * (long long)dst_lat_stride + (long long)col_4))[0] = reinterpret_cast<int2*>(packed_4)[0];
                }
                int col_5 = 128 + lane_2 * 4;
                float v4_3_1[4];
                v4_3_1[0] = mul_f32_rn_noftz((float)n_1[4], s_lat_1);
                if (bad_1 != 0) {
                    v4_3_1[0] = nan_1;
                }
                v4_3_1[1] = mul_f32_rn_noftz((float)n_1[5], s_lat_1);
                if (bad_1 != 0) {
                    v4_3_1[1] = nan_1;
                }
                v4_3_1[2] = mul_f32_rn_noftz((float)n_1[6], s_lat_1);
                if (bad_1 != 0) {
                    v4_3_1[2] = nan_1;
                }
                v4_3_1[3] = mul_f32_rn_noftz((float)n_1[7], s_lat_1);
                if (bad_1 != 0) {
                    v4_3_1[3] = nan_1;
                }
                if (accumulate != 0) {
                    if (touched_1 != 0) {
                        if (n_1[4] != 0 || bad_1 != 0) {
                            float _add_44;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(prev_1[4]), "f"(v4_3_1[0]));
                            prev_1[4] = _add_44;
                        }
                        if (n_1[5] != 0 || bad_1 != 0) {
                            float _add_45;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(prev_1[5]), "f"(v4_3_1[1]));
                            prev_1[5] = _add_45;
                        }
                        if (n_1[6] != 0 || bad_1 != 0) {
                            float _add_46;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(prev_1[6]), "f"(v4_3_1[2]));
                            prev_1[6] = _add_46;
                        }
                        if (n_1[7] != 0 || bad_1 != 0) {
                            float _add_47;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(prev_1[7]), "f"(v4_3_1[3]));
                            prev_1[7] = _add_47;
                        }
                        {
                            float4 _v4 = make_float4(prev_1[4 + 0], prev_1[4 + 1], prev_1[4 + 2], prev_1[4 + 3]);
                            *reinterpret_cast<float4*>(dst_packed + (long long)r * (long long)dst_row_stride + (long long)col_5) = _v4;
                        }
                    }
                } else if (out_f32 != 0) {
                    {
                        float4 _v4 = make_float4(v4_3_1[0 + 0], v4_3_1[0 + 1], v4_3_1[0 + 2], v4_3_1[0 + 3]);
                        *reinterpret_cast<float4*>(dst_latent_f32 + (long long)r * (long long)dst_lat_stride + (long long)col_5) = _v4;
                    }
                } else {
                    unsigned int packed_5[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_3_1[_lp*2 + 0], v4_3_1[_lp*2+1 + 0]));
                        packed_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int2*>(dst_latent + ((long long)r * (long long)dst_lat_stride + (long long)col_5))[0] = reinterpret_cast<int2*>(packed_5)[0];
                }
                int col_6 = 256 + lane_2 * 4;
                float v4_4[4];
                v4_4[0] = mul_f32_rn_noftz((float)n_1[8], s_lat_1);
                if (bad_1 != 0) {
                    v4_4[0] = nan_1;
                }
                v4_4[1] = mul_f32_rn_noftz((float)n_1[9], s_lat_1);
                if (bad_1 != 0) {
                    v4_4[1] = nan_1;
                }
                v4_4[2] = mul_f32_rn_noftz((float)n_1[10], s_lat_1);
                if (bad_1 != 0) {
                    v4_4[2] = nan_1;
                }
                v4_4[3] = mul_f32_rn_noftz((float)n_1[11], s_lat_1);
                if (bad_1 != 0) {
                    v4_4[3] = nan_1;
                }
                if (accumulate != 0) {
                    if (touched_1 != 0) {
                        if (n_1[8] != 0 || bad_1 != 0) {
                            float _add_48;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(prev_1[8]), "f"(v4_4[0]));
                            prev_1[8] = _add_48;
                        }
                        if (n_1[9] != 0 || bad_1 != 0) {
                            float _add_49;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(prev_1[9]), "f"(v4_4[1]));
                            prev_1[9] = _add_49;
                        }
                        if (n_1[10] != 0 || bad_1 != 0) {
                            float _add_50;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(prev_1[10]), "f"(v4_4[2]));
                            prev_1[10] = _add_50;
                        }
                        if (n_1[11] != 0 || bad_1 != 0) {
                            float _add_51;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(prev_1[11]), "f"(v4_4[3]));
                            prev_1[11] = _add_51;
                        }
                        {
                            float4 _v4 = make_float4(prev_1[8 + 0], prev_1[8 + 1], prev_1[8 + 2], prev_1[8 + 3]);
                            *reinterpret_cast<float4*>(dst_packed + (long long)r * (long long)dst_row_stride + (long long)col_6) = _v4;
                        }
                    }
                } else if (out_f32 != 0) {
                    {
                        float4 _v4 = make_float4(v4_4[0 + 0], v4_4[0 + 1], v4_4[0 + 2], v4_4[0 + 3]);
                        *reinterpret_cast<float4*>(dst_latent_f32 + (long long)r * (long long)dst_lat_stride + (long long)col_6) = _v4;
                    }
                } else {
                    unsigned int packed_6[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_4[_lp*2 + 0], v4_4[_lp*2+1 + 0]));
                        packed_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int2*>(dst_latent + ((long long)r * (long long)dst_lat_stride + (long long)col_6))[0] = reinterpret_cast<int2*>(packed_6)[0];
                }
                int col_7 = 384 + lane_2 * 4;
                float v4_5[4];
                v4_5[0] = mul_f32_rn_noftz((float)n_1[12], s_lat_1);
                if (bad_1 != 0) {
                    v4_5[0] = nan_1;
                }
                v4_5[1] = mul_f32_rn_noftz((float)n_1[13], s_lat_1);
                if (bad_1 != 0) {
                    v4_5[1] = nan_1;
                }
                v4_5[2] = mul_f32_rn_noftz((float)n_1[14], s_lat_1);
                if (bad_1 != 0) {
                    v4_5[2] = nan_1;
                }
                v4_5[3] = mul_f32_rn_noftz((float)n_1[15], s_lat_1);
                if (bad_1 != 0) {
                    v4_5[3] = nan_1;
                }
                if (accumulate != 0) {
                    if (touched_1 != 0) {
                        if (n_1[12] != 0 || bad_1 != 0) {
                            float _add_52;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(prev_1[12]), "f"(v4_5[0]));
                            prev_1[12] = _add_52;
                        }
                        if (n_1[13] != 0 || bad_1 != 0) {
                            float _add_53;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(prev_1[13]), "f"(v4_5[1]));
                            prev_1[13] = _add_53;
                        }
                        if (n_1[14] != 0 || bad_1 != 0) {
                            float _add_54;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(prev_1[14]), "f"(v4_5[2]));
                            prev_1[14] = _add_54;
                        }
                        if (n_1[15] != 0 || bad_1 != 0) {
                            float _add_55;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(prev_1[15]), "f"(v4_5[3]));
                            prev_1[15] = _add_55;
                        }
                        {
                            float4 _v4 = make_float4(prev_1[12 + 0], prev_1[12 + 1], prev_1[12 + 2], prev_1[12 + 3]);
                            *reinterpret_cast<float4*>(dst_packed + (long long)r * (long long)dst_row_stride + (long long)col_7) = _v4;
                        }
                    }
                } else if (out_f32 != 0) {
                    {
                        float4 _v4 = make_float4(v4_5[0 + 0], v4_5[0 + 1], v4_5[0 + 2], v4_5[0 + 3]);
                        *reinterpret_cast<float4*>(dst_latent_f32 + (long long)r * (long long)dst_lat_stride + (long long)col_7) = _v4;
                    }
                } else {
                    unsigned int packed_7[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_5[_lp*2 + 0], v4_5[_lp*2+1 + 0]));
                        packed_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int2*>(dst_latent + ((long long)r * (long long)dst_lat_stride + (long long)col_7))[0] = reinterpret_cast<int2*>(packed_7)[0];
                }
            }
        }
    }
}

} // extern "C"
