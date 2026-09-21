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

__global__ __launch_bounds__(256) void
kernel_cake_kda_chunk_train_d086ff5b482109313f4f(float* __restrict__ dq_intra, float* __restrict__ dk_intra, __nv_bfloat16* __restrict__ q_norm, __nv_bfloat16* __restrict__ k_norm, float* __restrict__ q_rstd, float* __restrict__ k_rstd, __nv_bfloat16* __restrict__ dq_out, __nv_bfloat16* __restrict__ dk_out, int num_qk_heads, int num_v_heads, int group)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    long long row0 = (long long)chunk * 64;
    int warp_0 = warp;
    int d0 = lane * 4;
    #pragma unroll
    for (int it = 0; it < 8; it++) {
        long long row = row0 + (long long)(it * 8 + warp_0);
        float dyq[4];
        float dyk[4];
        dyq[0] = 0.0f;
        dyq[1] = 0.0f;
        dyq[2] = 0.0f;
        dyq[3] = 0.0f;
        dyk[0] = 0.0f;
        dyk[1] = 0.0f;
        dyk[2] = 0.0f;
        dyk[3] = 0.0f;
        #pragma unroll 1
        for (int g = 0; g < group; g++) {
            long long vindex = (row * (long long)num_v_heads + (long long)(head * group + g)) * 128 + (long long)d0;
            float tq[4];
            float tk[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dq_intra + vindex);
                tq[0 + 0] = _v4.x;
                tq[0 + 1] = _v4.y;
                tq[0 + 2] = _v4.z;
                tq[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dk_intra + vindex);
                tk[0 + 0] = _v4.x;
                tk[0 + 1] = _v4.y;
                tk[0 + 2] = _v4.z;
                tk[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                dyq[e] = dyq[e] + tq[e];
                dyk[e] = dyk[e] + tk[e];
            }
        }
        long long qindex = (row * (long long)num_qk_heads + (long long)head) * 128 + (long long)d0;
        float yq[4];
        float yk[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(q_norm + qindex);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yq[0 + _pair * 2])[0]), "=f"((&yq[0 + _pair * 2])[1])
                    : "r"(_vpairs_2[_pair]));
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(k_norm + qindex);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&yk[0 + _pair * 2])[0]), "=f"((&yk[0 + _pair * 2])[1])
                    : "r"(_vpairs_3[_pair]));
            }
        }
        float pq = 0.0f;
        float pk = 0.0f;
        #pragma unroll
        for (int e_1 = 0; e_1 < 4; e_1++) {
            float _fma_0 = __fmaf_rn(dyq[e_1], yq[e_1], pq);
            pq = _fma_0;
            float _fma_1 = __fmaf_rn(dyk[e_1], yk[e_1], pk);
            pk = _fma_1;
        }
        float _warp_reduce_0 = pq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        float dot_q = _warp_reduce_0;
        float _warp_reduce_1 = pk;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        float dot_k = _warp_reduce_1;
        long long rstd_index = row * (long long)num_qk_heads + (long long)head;
        float rq = q_rstd[rstd_index];
        float rk = k_rstd[rstd_index];
        float oq[4];
        float ok[4];
        #pragma unroll
        for (int e_2 = 0; e_2 < 4; e_2++) {
            oq[e_2] = dyq[e_2] * rq - dot_q * yq[e_2] * rq;
            ok[e_2] = dyk[e_2] * rk - dot_k * yk[e_2] * rk;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(oq[0 + 0], oq[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(oq[0 + 2], oq[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dq_out + qindex))[0]) = _pk2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(ok[0 + 0], ok[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(ok[0 + 2], ok[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dk_out + qindex))[0]) = _pk2;
        }
    }
}

} // extern "C"
