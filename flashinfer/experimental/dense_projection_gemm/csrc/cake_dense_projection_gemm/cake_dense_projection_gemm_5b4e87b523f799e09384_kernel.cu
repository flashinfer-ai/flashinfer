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

#include <math_constants.h>


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_dense_projection_gemm_5b4e87b523f799e09384(float* __restrict__ src, __nv_bfloat16* __restrict__ parts, int n)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    unsigned long long nn = (unsigned long long)n;
    unsigned long long e0 = (unsigned long long)(bid * 256 + tid) * 8;
    unsigned long long e1 = e0 + 4;
    int full = ((nn >= e0 + 8) ? 1 : 0);
    float x[8];
    float hi[8];
    float mid[8];
    float lo[8];
    if (e0 < nn) {
        unsigned long long src_bits = (unsigned long long)(src + e0);
        int src_vec = full;
        if ((src_bits & 15) != 0) {
            src_vec = 0;
        }
        if (src_vec == 1) {
            float _vec_load_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(src + e0 + 0);
                _vec_load_0[0 + 0] = _v4.x;
                _vec_load_0[0 + 1] = _v4.y;
                _vec_load_0[0 + 2] = _v4.z;
                _vec_load_0[0 + 3] = _v4.w;
            }
            float _vec_load_1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(src + e1 + 0);
                _vec_load_1[0 + 0] = _v4.x;
                _vec_load_1[0 + 1] = _v4.y;
                _vec_load_1[0 + 2] = _v4.z;
                _vec_load_1[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                x[j] = _vec_load_0[j];
                x[4 + j] = _vec_load_1[j];
            }
        } else {
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                x[j_1] = 0.0f;
                if (nn > e0 + (unsigned long long)j_1) {
                    x[j_1] = src[e0 + (unsigned long long)j_1];
                }
            }
        }
        #pragma unroll
        for (int j_2 = 0; j_2 < 4; j_2++) {
            float v_a = x[2 * j_2];
            float v_b = x[2 * j_2 + 1];
            unsigned int pk = __as_u32(__float22bfloat162_rn(make_float2(v_a, v_b)));
            float h_a = __uint_as_float(pk << 16);
            float h_b = __uint_as_float(pk & 4294901760u);
            float r_a = v_a - h_a;
            float r_b = v_b - h_b;
            unsigned int pm = __as_u32(__float22bfloat162_rn(make_float2(r_a, r_b)));
            float m_a = __uint_as_float(pm << 16);
            float m_b = __uint_as_float(pm & 4294901760u);
            hi[2 * j_2] = h_a;
            hi[2 * j_2 + 1] = h_b;
            mid[2 * j_2] = m_a;
            mid[2 * j_2 + 1] = m_b;
            lo[2 * j_2] = r_a - m_a;
            lo[2 * j_2 + 1] = r_b - m_b;
        }
        unsigned long long pk_off_1_0 = e0;
        unsigned long long part_bits_2_0 = (unsigned long long)(parts + pk_off_1_0);
        int part_vec_1_0 = full;
        if ((part_bits_2_0 & 15) != 0) {
            part_vec_1_0 = 0;
        }
        if (part_vec_1_0 == 1) {
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi[_lp*2 + 0], hi[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi[_lp*2 + 2], hi[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi[_lp*2 + 4], hi[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi[_lp*2 + 6], hi[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(parts + pk_off_1_0)[0] = reinterpret_cast<int4*>(_packed_0)[0];
        } else {
            #pragma unroll
            for (int j_1_0 = 0; j_1_0 < 8; j_1_0++) {
                if (nn > e0 + (unsigned long long)j_1_0) {
                    *(reinterpret_cast<__nv_bfloat16*>(parts + (pk_off_1_0 + (unsigned long long)j_1_0)) + (0)) = (__nv_bfloat16)hi[j_1_0];
                }
            }
        }
        unsigned long long pk_off_1_1 = nn + e0;
        unsigned long long part_bits_2_1 = (unsigned long long)(parts + pk_off_1_1);
        int part_vec_1_1 = full;
        if ((part_bits_2_1 & 15) != 0) {
            part_vec_1_1 = 0;
        }
        if (part_vec_1_1 == 1) {
            unsigned int _packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mid[_lp*2 + 0], mid[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mid[_lp*2 + 2], mid[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mid[_lp*2 + 4], mid[_lp*2+1 + 4]));
                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mid[_lp*2 + 6], mid[_lp*2+1 + 6]));
                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(parts + pk_off_1_1)[0] = reinterpret_cast<int4*>(_packed_1)[0];
        } else {
            #pragma unroll
            for (int j_1_1 = 0; j_1_1 < 8; j_1_1++) {
                if (nn > e0 + (unsigned long long)j_1_1) {
                    *(reinterpret_cast<__nv_bfloat16*>(parts + (pk_off_1_1 + (unsigned long long)j_1_1)) + (0)) = (__nv_bfloat16)mid[j_1_1];
                }
            }
        }
        unsigned long long pk_off_1_2 = nn * 2 + e0;
        unsigned long long part_bits_2_2 = (unsigned long long)(parts + pk_off_1_2);
        int part_vec_1_2 = full;
        if ((part_bits_2_2 & 15) != 0) {
            part_vec_1_2 = 0;
        }
        if (part_vec_1_2 == 1) {
            unsigned int _packed_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(parts + pk_off_1_2)[0] = reinterpret_cast<int4*>(_packed_2)[0];
        } else {
            #pragma unroll
            for (int j_1_2 = 0; j_1_2 < 8; j_1_2++) {
                if (nn > e0 + (unsigned long long)j_1_2) {
                    *(reinterpret_cast<__nv_bfloat16*>(parts + (pk_off_1_2 + (unsigned long long)j_1_2)) + (0)) = (__nv_bfloat16)lo[j_1_2];
                }
            }
        }
    }
}

} // extern "C"
