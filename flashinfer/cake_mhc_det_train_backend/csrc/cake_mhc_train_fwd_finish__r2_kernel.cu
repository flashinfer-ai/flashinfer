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
#define LAUNCH_BOUNDS_MIN_BLOCKS 2
#define LAUNCH_MIN_BLOCKS 2

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

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS, LAUNCH_BOUNDS_MIN_BLOCKS) void
kernel_cake_mhc_train_fwd_finish__r2(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ b, float* __restrict__ stats, __nv_bfloat16* __restrict__ y, int T)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int group = tid / 128;
    int gtid = tid - group * 128;
    long long col0 = (long long)(gtid * 16);
    int nsub = T * 2;
    #pragma unroll 1
    for (int row0 = bid * 2; row0 < nsub; row0 += num_bids * 2) {
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
            unsigned int cache[40];
            #pragma unroll
            for (int v = 0; v < 1; v++) {
                {
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(cache[v * 8 + 0]), "=r"(cache[v * 8 + 1]), "=r"(cache[v * 8 + 2]), "=r"(cache[v * 8 + 3]), "=r"(cache[v * 8 + 4]), "=r"(cache[v * 8 + 5]), "=r"(cache[v * 8 + 6]), "=r"(cache[v * 8 + 7]) : "l"((const void*)((const char*)(b + (bbase + (long long)(v * 2048)) + 0) + 0)) : "memory");
                }
            }
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                #pragma unroll
                for (int v_1 = 0; v_1 < 1; v_1++) {
                    {
                        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(cache[8 + (s + v_1) * 8 + 0]), "=r"(cache[8 + (s + v_1) * 8 + 1]), "=r"(cache[8 + (s + v_1) * 8 + 2]), "=r"(cache[8 + (s + v_1) * 8 + 3]), "=r"(cache[8 + (s + v_1) * 8 + 4]), "=r"(cache[8 + (s + v_1) * 8 + 5]), "=r"(cache[8 + (s + v_1) * 8 + 6]), "=r"(cache[8 + (s + v_1) * 8 + 7]) : "l"((const void*)((const char*)(x + (xbase + (long long)(s * 4096 + v_1 * 2048)) + 0) + 0)) : "memory");
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                #pragma unroll
                for (int v_2 = 0; v_2 < 1; v_2++) {
                    unsigned int yw[8];
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        float _cvt_f32_bf16_4;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_4) : "h"((uint16_t)(cache[(4 + v_2) * 8 + k])));
                        float _cvt_f32_bf16_6;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_6) : "h"((uint16_t)(cache[(3 + v_2) * 8 + k])));
                        float _cvt_f32_bf16_8;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_8) : "h"((uint16_t)(cache[(2 + v_2) * 8 + k])));
                        float _cvt_f32_bf16_10;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_10) : "h"((uint16_t)(cache[(1 + v_2) * 8 + k])));
                        float _cvt_f32_bf16_11;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_11) : "h"((uint16_t)(cache[v_2 * 8 + k])));
                        float _fma_9;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(coef[5 + j]), "f"(_cvt_f32_bf16_10), "f"(mul_f32_rn_noftz(coef[1 + j], _cvt_f32_bf16_11)));
                        float _fma_7;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(coef[9 + j]), "f"(_cvt_f32_bf16_8), "f"(_fma_9));
                        float _fma_5;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(coef[13 + j]), "f"(_cvt_f32_bf16_6), "f"(_fma_7));
                        float _fma_3;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(coef[17 + j]), "f"(_cvt_f32_bf16_4), "f"(_fma_5));
                        uint32_t _prmt_14;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_14) : "r"(cache[(4 + v_2) * 8 + k]), "r"(0));
                        float _cvt_f32_bf16_13;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_13) : "h"((uint16_t)(_prmt_14)));
                        uint32_t _prmt_17;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_17) : "r"(cache[(3 + v_2) * 8 + k]), "r"(0));
                        float _cvt_f32_bf16_16;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_16) : "h"((uint16_t)(_prmt_17)));
                        uint32_t _prmt_20;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_20) : "r"(cache[(2 + v_2) * 8 + k]), "r"(0));
                        float _cvt_f32_bf16_19;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_19) : "h"((uint16_t)(_prmt_20)));
                        uint32_t _prmt_23;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_23) : "r"(cache[(1 + v_2) * 8 + k]), "r"(0));
                        float _cvt_f32_bf16_22;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_22) : "h"((uint16_t)(_prmt_23)));
                        uint32_t _prmt_25;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_25) : "r"(cache[v_2 * 8 + k]), "r"(0));
                        float _cvt_f32_bf16_24;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_24) : "h"((uint16_t)(_prmt_25)));
                        float _fma_21;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(coef[5 + j]), "f"(_cvt_f32_bf16_22), "f"(mul_f32_rn_noftz(coef[1 + j], _cvt_f32_bf16_24)));
                        float _fma_18;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(coef[9 + j]), "f"(_cvt_f32_bf16_19), "f"(_fma_21));
                        float _fma_15;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(coef[13 + j]), "f"(_cvt_f32_bf16_16), "f"(_fma_18));
                        float _fma_12;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(coef[17 + j]), "f"(_cvt_f32_bf16_13), "f"(_fma_15));
                        unsigned int hword = __as_u32(__float22bfloat162_rn(make_float2(_fma_3, _fma_12)));
                        yw[k] = hword;
                    }
                    {
                        const unsigned* _raw_stv8_26 = reinterpret_cast<const unsigned*>(yw);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(reinterpret_cast<unsigned int*>(y + (xbase + (long long)(j * 4096 + v_2 * 2048))))), "r"(_raw_stv8_26[0]), "r"(_raw_stv8_26[1]), "r"(_raw_stv8_26[2]), "r"(_raw_stv8_26[3]), "r"(_raw_stv8_26[4]), "r"(_raw_stv8_26[5]), "r"(_raw_stv8_26[6]), "r"(_raw_stv8_26[7]) : "memory");
                    }
                }
            }
        }
    }
}

} // extern "C"
