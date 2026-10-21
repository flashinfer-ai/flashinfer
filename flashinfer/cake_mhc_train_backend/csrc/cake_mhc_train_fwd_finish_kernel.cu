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
#define LAUNCH_BOUNDS_THREADS 256

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

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_mhc_train_fwd_finish(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ b, float* __restrict__ stats, __nv_bfloat16* __restrict__ y, int T)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int group = tid / 64;
    int gtid = tid - group * 64;
    long long col0 = (long long)(gtid * 16);
    int nsub = T * 2;
    #pragma unroll 1
    for (int row0 = bid * 4; row0 < nsub; row0 += num_bids * 4) {
        int sub = row0 + group;
        if (sub < nsub) {
            int row = sub / 2;
            int part = sub - row * 2;
            long long row64 = (long long)row;
            long long colb = (long long)(part * 2048) + col0;
            long long xbase = row64 * 16384 + colb;
            long long bbase = row64 * 4096 + colb;
            long long sbase = row64 * 64 + 28;
            float coef[24];
            #pragma unroll
            for (int q = 0; q < 6; q++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(stats + (sbase + (long long)(4 * q)) + 0);
                    coef[4 * q + 0] = _v4.x;
                    coef[4 * q + 1] = _v4.y;
                    coef[4 * q + 2] = _v4.z;
                    coef[4 * q + 3] = _v4.w;
                }
            }
            unsigned int cache[80];
            #pragma unroll
            for (int v = 0; v < 2; v++) {
                {
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(cache[v * 8 + 0]), "=r"(cache[v * 8 + 1]), "=r"(cache[v * 8 + 2]), "=r"(cache[v * 8 + 3]), "=r"(cache[v * 8 + 4]), "=r"(cache[v * 8 + 5]), "=r"(cache[v * 8 + 6]), "=r"(cache[v * 8 + 7]) : "l"((const void*)((const char*)(b + (bbase + (long long)(v * 1024)) + 0) + 0)) : "memory");
                }
            }
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                #pragma unroll
                for (int v_1 = 0; v_1 < 2; v_1++) {
                    {
                        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(cache[16 + (s * 2 + v_1) * 8 + 0]), "=r"(cache[16 + (s * 2 + v_1) * 8 + 1]), "=r"(cache[16 + (s * 2 + v_1) * 8 + 2]), "=r"(cache[16 + (s * 2 + v_1) * 8 + 3]), "=r"(cache[16 + (s * 2 + v_1) * 8 + 4]), "=r"(cache[16 + (s * 2 + v_1) * 8 + 5]), "=r"(cache[16 + (s * 2 + v_1) * 8 + 6]), "=r"(cache[16 + (s * 2 + v_1) * 8 + 7]) : "l"((const void*)((const char*)(x + (xbase + (long long)(s * 4096 + v_1 * 1024)) + 0) + 0)) : "memory");
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                #pragma unroll
                for (int v_2 = 0; v_2 < 2; v_2++) {
                    unsigned int yw[8];
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(cache[v_2 * 8 + k]));
                        float _mul_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(coef[1 + j]), "f"(_cvt_f32_0.x));
                        float _mul_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(coef[1 + j]), "f"(_cvt_f32_0.y));
                        float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(cache[(2 + v_2) * 8 + k]));
                        float _fma_0;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(coef[5 + j]), "f"(_cvt_f32_1.x), "f"(_mul_0));
                        float _fma_1;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(coef[5 + j]), "f"(_cvt_f32_1.y), "f"(_mul_1));
                        float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(cache[(4 + v_2) * 8 + k]));
                        float _fma_2;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(coef[9 + j]), "f"(_cvt_f32_2.x), "f"(_fma_0));
                        float _fma_3;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(coef[9 + j]), "f"(_cvt_f32_2.y), "f"(_fma_1));
                        float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(cache[(6 + v_2) * 8 + k]));
                        float _fma_4;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(coef[13 + j]), "f"(_cvt_f32_3.x), "f"(_fma_2));
                        float _fma_5;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(coef[13 + j]), "f"(_cvt_f32_3.y), "f"(_fma_3));
                        float2 _cvt_f32_4 = __bfloat1622float2(__as_bf16x2(cache[(8 + v_2) * 8 + k]));
                        float _fma_6;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(coef[17 + j]), "f"(_cvt_f32_4.x), "f"(_fma_4));
                        float _fma_7;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(coef[17 + j]), "f"(_cvt_f32_4.y), "f"(_fma_5));
                        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_fma_6, _fma_7));
                        unsigned int hword = __as_u32(_bf16x2_0);
                        yw[k] = hword;
                    }
                    asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(y + (xbase + (long long)(j * 4096 + v_2 * 1024))) + (0)), "r"(yw[(0) + 0]), "r"(yw[(0) + 1]), "r"(yw[(0) + 2]), "r"(yw[(0) + 3]), "r"(yw[(0) + 4]), "r"(yw[(0) + 5]), "r"(yw[(0) + 6]), "r"(yw[(0) + 7]) : "memory");
                }
            }
        }
    }
}

} // extern "C"
