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
kernel_cake_dsa_h64_det_train_9a9e7e6b8ea3cd322457(int* dkv_acc, int* dkr_acc, int* __restrict__ e_lat, int* __restrict__ e_rope, unsigned long long* __restrict__ cnt64, unsigned int* __restrict__ row_pairs, __nv_bfloat16* dst_latent, __nv_bfloat16* dst_rope, float* dst_latent_f32, float* dst_rope_f32, float* dst_packed, int num_rows, int dkv_stride, int dkr_stride, int dkr_col0, int dst_lat_stride, int dst_rope_stride, int out_f32, int dst_row_stride, int accumulate)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int row = blockIdx.x * 8 + warp;
    int lane_1 = lane;
    if (row < num_rows) {
        int word0 = e_lat[row];
        int e = word0 << 16 >> 16;
        int touched = (int)(e != 127);
        int bad = 0;
        if (touched != 0) {
            unsigned long long worst = 0ull;
            unsigned long long cq_0 = cnt64[(long long)row * 4ll];
            unsigned long long _max_0 = ((worst) > (cq_0) ? (worst) : (cq_0));
            worst = _max_0;
            unsigned long long cq_1 = cnt64[(long long)row * 4ll + 1ll];
            unsigned long long _max_1 = ((worst) > (cq_1) ? (worst) : (cq_1));
            worst = _max_1;
            unsigned long long cq_2 = cnt64[(long long)row * 4ll + 2ll];
            unsigned long long _max_2 = ((worst) > (cq_2) ? (worst) : (cq_2));
            worst = _max_2;
            unsigned long long cq_3 = cnt64[(long long)row * 4ll + 3ll];
            unsigned long long _max_3 = ((worst) > (cq_3) ? (worst) : (cq_3));
            worst = _max_3;
            unsigned long long slack = (unsigned long long)row_pairs[row] << (unsigned long long)(word0 >> 16);
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
        int n4[4];
        float v4[4];
        if (touched != 0) {
            {
                const int4* _ivptr_4 = reinterpret_cast<const int4*>(dkv_acc + ((long long)row * (long long)dkv_stride + (long long)col_0) + 0);
                int4 _ivld_4;
                _ivld_4 = *_ivptr_4;
                n4[0 + 0] = _ivld_4.x;
                n4[0 + 1] = _ivld_4.y;
                n4[0 + 2] = _ivld_4.z;
                n4[0 + 3] = _ivld_4.w;
            }
        } else {
            n4[0] = 0;
            n4[1] = 0;
            n4[2] = 0;
            n4[3] = 0;
        }
        v4[0] = mul_f32_rn_noftz((float)n4[0], s_lat);
        if (bad != 0) {
            v4[0] = nan;
        }
        v4[1] = mul_f32_rn_noftz((float)n4[1], s_lat);
        if (bad != 0) {
            v4[1] = nan;
        }
        v4[2] = mul_f32_rn_noftz((float)n4[2], s_lat);
        if (bad != 0) {
            v4[2] = nan;
        }
        v4[3] = mul_f32_rn_noftz((float)n4[3], s_lat);
        if (bad != 0) {
            v4[3] = nan;
        }
        if (accumulate != 0) {
            if (touched != 0) {
                float prev4[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)col_0) + 0);
                    prev4[0 + 0] = _v4.x;
                    prev4[0 + 1] = _v4.y;
                    prev4[0 + 2] = _v4.z;
                    prev4[0 + 3] = _v4.w;
                }
                if (n4[0] != 0 || bad != 0) {
                    float _add_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(prev4[0]), "f"(v4[0]));
                    prev4[0] = _add_6;
                }
                if (n4[1] != 0 || bad != 0) {
                    float _add_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(prev4[1]), "f"(v4[1]));
                    prev4[1] = _add_7;
                }
                if (n4[2] != 0 || bad != 0) {
                    float _add_8;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(prev4[2]), "f"(v4[2]));
                    prev4[2] = _add_8;
                }
                if (n4[3] != 0 || bad != 0) {
                    float _add_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(prev4[3]), "f"(v4[3]));
                    prev4[3] = _add_9;
                }
                {
                    float4 _v4 = make_float4(prev4[0 + 0], prev4[0 + 1], prev4[0 + 2], prev4[0 + 3]);
                    *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)col_0) = _v4;
                }
            }
        } else if (out_f32 != 0) {
            {
                float4 _v4 = make_float4(v4[0 + 0], v4[0 + 1], v4[0 + 2], v4[0 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + (long long)row * (long long)dst_lat_stride + (long long)col_0) = _v4;
            }
        } else {
            unsigned int packed[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4[_lp*2 + 0], v4[_lp*2+1 + 0]));
                packed[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dst_latent + ((long long)row * (long long)dst_lat_stride + (long long)col_0))[0] = reinterpret_cast<int2*>(packed)[0];
        }
        int col_1 = 128 + lane_1 * 4;
        int n4_0[4];
        float v4_0[4];
        if (touched != 0) {
            {
                const int4* _ivptr_10 = reinterpret_cast<const int4*>(dkv_acc + ((long long)row * (long long)dkv_stride + (long long)col_1) + 0);
                int4 _ivld_10;
                _ivld_10 = *_ivptr_10;
                n4_0[0 + 0] = _ivld_10.x;
                n4_0[0 + 1] = _ivld_10.y;
                n4_0[0 + 2] = _ivld_10.z;
                n4_0[0 + 3] = _ivld_10.w;
            }
        } else {
            n4_0[0] = 0;
            n4_0[1] = 0;
            n4_0[2] = 0;
            n4_0[3] = 0;
        }
        v4_0[0] = mul_f32_rn_noftz((float)n4_0[0], s_lat);
        if (bad != 0) {
            v4_0[0] = nan;
        }
        v4_0[1] = mul_f32_rn_noftz((float)n4_0[1], s_lat);
        if (bad != 0) {
            v4_0[1] = nan;
        }
        v4_0[2] = mul_f32_rn_noftz((float)n4_0[2], s_lat);
        if (bad != 0) {
            v4_0[2] = nan;
        }
        v4_0[3] = mul_f32_rn_noftz((float)n4_0[3], s_lat);
        if (bad != 0) {
            v4_0[3] = nan;
        }
        if (accumulate != 0) {
            if (touched != 0) {
                float prev4_1[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)col_1) + 0);
                    prev4_1[0 + 0] = _v4.x;
                    prev4_1[0 + 1] = _v4.y;
                    prev4_1[0 + 2] = _v4.z;
                    prev4_1[0 + 3] = _v4.w;
                }
                if (n4_0[0] != 0 || bad != 0) {
                    float _add_12;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(prev4_1[0]), "f"(v4_0[0]));
                    prev4_1[0] = _add_12;
                }
                if (n4_0[1] != 0 || bad != 0) {
                    float _add_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(prev4_1[1]), "f"(v4_0[1]));
                    prev4_1[1] = _add_13;
                }
                if (n4_0[2] != 0 || bad != 0) {
                    float _add_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(prev4_1[2]), "f"(v4_0[2]));
                    prev4_1[2] = _add_14;
                }
                if (n4_0[3] != 0 || bad != 0) {
                    float _add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(prev4_1[3]), "f"(v4_0[3]));
                    prev4_1[3] = _add_15;
                }
                {
                    float4 _v4 = make_float4(prev4_1[0 + 0], prev4_1[0 + 1], prev4_1[0 + 2], prev4_1[0 + 3]);
                    *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)col_1) = _v4;
                }
            }
        } else if (out_f32 != 0) {
            {
                float4 _v4 = make_float4(v4_0[0 + 0], v4_0[0 + 1], v4_0[0 + 2], v4_0[0 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + (long long)row * (long long)dst_lat_stride + (long long)col_1) = _v4;
            }
        } else {
            unsigned int packed_1[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_0[_lp*2 + 0], v4_0[_lp*2+1 + 0]));
                packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dst_latent + ((long long)row * (long long)dst_lat_stride + (long long)col_1))[0] = reinterpret_cast<int2*>(packed_1)[0];
        }
        int col_2 = 256 + lane_1 * 4;
        int n4_1[4];
        float v4_1[4];
        if (touched != 0) {
            {
                const int4* _ivptr_16 = reinterpret_cast<const int4*>(dkv_acc + ((long long)row * (long long)dkv_stride + (long long)col_2) + 0);
                int4 _ivld_16;
                _ivld_16 = *_ivptr_16;
                n4_1[0 + 0] = _ivld_16.x;
                n4_1[0 + 1] = _ivld_16.y;
                n4_1[0 + 2] = _ivld_16.z;
                n4_1[0 + 3] = _ivld_16.w;
            }
        } else {
            n4_1[0] = 0;
            n4_1[1] = 0;
            n4_1[2] = 0;
            n4_1[3] = 0;
        }
        v4_1[0] = mul_f32_rn_noftz((float)n4_1[0], s_lat);
        if (bad != 0) {
            v4_1[0] = nan;
        }
        v4_1[1] = mul_f32_rn_noftz((float)n4_1[1], s_lat);
        if (bad != 0) {
            v4_1[1] = nan;
        }
        v4_1[2] = mul_f32_rn_noftz((float)n4_1[2], s_lat);
        if (bad != 0) {
            v4_1[2] = nan;
        }
        v4_1[3] = mul_f32_rn_noftz((float)n4_1[3], s_lat);
        if (bad != 0) {
            v4_1[3] = nan;
        }
        if (accumulate != 0) {
            if (touched != 0) {
                float prev4_2[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)col_2) + 0);
                    prev4_2[0 + 0] = _v4.x;
                    prev4_2[0 + 1] = _v4.y;
                    prev4_2[0 + 2] = _v4.z;
                    prev4_2[0 + 3] = _v4.w;
                }
                if (n4_1[0] != 0 || bad != 0) {
                    float _add_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(prev4_2[0]), "f"(v4_1[0]));
                    prev4_2[0] = _add_18;
                }
                if (n4_1[1] != 0 || bad != 0) {
                    float _add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(prev4_2[1]), "f"(v4_1[1]));
                    prev4_2[1] = _add_19;
                }
                if (n4_1[2] != 0 || bad != 0) {
                    float _add_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(prev4_2[2]), "f"(v4_1[2]));
                    prev4_2[2] = _add_20;
                }
                if (n4_1[3] != 0 || bad != 0) {
                    float _add_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(prev4_2[3]), "f"(v4_1[3]));
                    prev4_2[3] = _add_21;
                }
                {
                    float4 _v4 = make_float4(prev4_2[0 + 0], prev4_2[0 + 1], prev4_2[0 + 2], prev4_2[0 + 3]);
                    *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)col_2) = _v4;
                }
            }
        } else if (out_f32 != 0) {
            {
                float4 _v4 = make_float4(v4_1[0 + 0], v4_1[0 + 1], v4_1[0 + 2], v4_1[0 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + (long long)row * (long long)dst_lat_stride + (long long)col_2) = _v4;
            }
        } else {
            unsigned int packed_2[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_1[_lp*2 + 0], v4_1[_lp*2+1 + 0]));
                packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dst_latent + ((long long)row * (long long)dst_lat_stride + (long long)col_2))[0] = reinterpret_cast<int2*>(packed_2)[0];
        }
        int col_3 = 384 + lane_1 * 4;
        int n4_2[4];
        float v4_2[4];
        if (touched != 0) {
            {
                const int4* _ivptr_22 = reinterpret_cast<const int4*>(dkv_acc + ((long long)row * (long long)dkv_stride + (long long)col_3) + 0);
                int4 _ivld_22;
                _ivld_22 = *_ivptr_22;
                n4_2[0 + 0] = _ivld_22.x;
                n4_2[0 + 1] = _ivld_22.y;
                n4_2[0 + 2] = _ivld_22.z;
                n4_2[0 + 3] = _ivld_22.w;
            }
        } else {
            n4_2[0] = 0;
            n4_2[1] = 0;
            n4_2[2] = 0;
            n4_2[3] = 0;
        }
        v4_2[0] = mul_f32_rn_noftz((float)n4_2[0], s_lat);
        if (bad != 0) {
            v4_2[0] = nan;
        }
        v4_2[1] = mul_f32_rn_noftz((float)n4_2[1], s_lat);
        if (bad != 0) {
            v4_2[1] = nan;
        }
        v4_2[2] = mul_f32_rn_noftz((float)n4_2[2], s_lat);
        if (bad != 0) {
            v4_2[2] = nan;
        }
        v4_2[3] = mul_f32_rn_noftz((float)n4_2[3], s_lat);
        if (bad != 0) {
            v4_2[3] = nan;
        }
        if (accumulate != 0) {
            if (touched != 0) {
                float prev4_3[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)col_3) + 0);
                    prev4_3[0 + 0] = _v4.x;
                    prev4_3[0 + 1] = _v4.y;
                    prev4_3[0 + 2] = _v4.z;
                    prev4_3[0 + 3] = _v4.w;
                }
                if (n4_2[0] != 0 || bad != 0) {
                    float _add_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(prev4_3[0]), "f"(v4_2[0]));
                    prev4_3[0] = _add_24;
                }
                if (n4_2[1] != 0 || bad != 0) {
                    float _add_25;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(prev4_3[1]), "f"(v4_2[1]));
                    prev4_3[1] = _add_25;
                }
                if (n4_2[2] != 0 || bad != 0) {
                    float _add_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(prev4_3[2]), "f"(v4_2[2]));
                    prev4_3[2] = _add_26;
                }
                if (n4_2[3] != 0 || bad != 0) {
                    float _add_27;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(prev4_3[3]), "f"(v4_2[3]));
                    prev4_3[3] = _add_27;
                }
                {
                    float4 _v4 = make_float4(prev4_3[0 + 0], prev4_3[0 + 1], prev4_3[0 + 2], prev4_3[0 + 3]);
                    *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)col_3) = _v4;
                }
            }
        } else if (out_f32 != 0) {
            {
                float4 _v4 = make_float4(v4_2[0 + 0], v4_2[0 + 1], v4_2[0 + 2], v4_2[0 + 3]);
                *reinterpret_cast<float4*>(dst_latent_f32 + (long long)row * (long long)dst_lat_stride + (long long)col_3) = _v4;
            }
        } else {
            unsigned int packed_3[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_2[_lp*2 + 0], v4_2[_lp*2+1 + 0]));
                packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dst_latent + ((long long)row * (long long)dst_lat_stride + (long long)col_3))[0] = reinterpret_cast<int2*>(packed_3)[0];
        }
        if (lane_1 < 16) {
            float s_rope = 0.0f;
            if (touched != 0) {
                s_rope = __uint_as_float((unsigned int)(e_rope[row] + 127) << 23);
            }
            int rcol = lane_1 * 4;
            int n4_3[4];
            float v4_3[4];
            if (touched != 0) {
                {
                    const int4* _ivptr_28 = reinterpret_cast<const int4*>(dkr_acc + ((long long)row * (long long)dkr_stride + (long long)(dkr_col0 + rcol)) + 0);
                    int4 _ivld_28;
                    _ivld_28 = *_ivptr_28;
                    n4_3[0 + 0] = _ivld_28.x;
                    n4_3[0 + 1] = _ivld_28.y;
                    n4_3[0 + 2] = _ivld_28.z;
                    n4_3[0 + 3] = _ivld_28.w;
                }
            } else {
                n4_3[0] = 0;
                n4_3[1] = 0;
                n4_3[2] = 0;
                n4_3[3] = 0;
            }
            v4_3[0] = mul_f32_rn_noftz((float)n4_3[0], s_rope);
            if (bad != 0) {
                v4_3[0] = nan;
            }
            v4_3[1] = mul_f32_rn_noftz((float)n4_3[1], s_rope);
            if (bad != 0) {
                v4_3[1] = nan;
            }
            v4_3[2] = mul_f32_rn_noftz((float)n4_3[2], s_rope);
            if (bad != 0) {
                v4_3[2] = nan;
            }
            v4_3[3] = mul_f32_rn_noftz((float)n4_3[3], s_rope);
            if (bad != 0) {
                v4_3[3] = nan;
            }
            if (accumulate != 0) {
                if (touched != 0) {
                    float prev4_4[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dst_packed + ((long long)row * (long long)dst_row_stride + (long long)(512 + rcol)) + 0);
                        prev4_4[0 + 0] = _v4.x;
                        prev4_4[0 + 1] = _v4.y;
                        prev4_4[0 + 2] = _v4.z;
                        prev4_4[0 + 3] = _v4.w;
                    }
                    if (n4_3[0] != 0 || bad != 0) {
                        float _add_30;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(prev4_4[0]), "f"(v4_3[0]));
                        prev4_4[0] = _add_30;
                    }
                    if (n4_3[1] != 0 || bad != 0) {
                        float _add_31;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(prev4_4[1]), "f"(v4_3[1]));
                        prev4_4[1] = _add_31;
                    }
                    if (n4_3[2] != 0 || bad != 0) {
                        float _add_32;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(prev4_4[2]), "f"(v4_3[2]));
                        prev4_4[2] = _add_32;
                    }
                    if (n4_3[3] != 0 || bad != 0) {
                        float _add_33;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(prev4_4[3]), "f"(v4_3[3]));
                        prev4_4[3] = _add_33;
                    }
                    {
                        float4 _v4 = make_float4(prev4_4[0 + 0], prev4_4[0 + 1], prev4_4[0 + 2], prev4_4[0 + 3]);
                        *reinterpret_cast<float4*>(dst_packed + (long long)row * (long long)dst_row_stride + (long long)(512 + rcol)) = _v4;
                    }
                }
            } else if (out_f32 != 0) {
                {
                    float4 _v4 = make_float4(v4_3[0 + 0], v4_3[0 + 1], v4_3[0 + 2], v4_3[0 + 3]);
                    *reinterpret_cast<float4*>(dst_rope_f32 + (long long)row * (long long)dst_rope_stride + (long long)rcol) = _v4;
                }
            } else {
                unsigned int packed_4[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(v4_3[_lp*2 + 0], v4_3[_lp*2+1 + 0]));
                    packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dst_rope + ((long long)row * (long long)dst_rope_stride + (long long)rcol))[0] = reinterpret_cast<int2*>(packed_4)[0];
            }
        }
    }
}

} // extern "C"
