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
kernel_cake_mhc_train_fwd_collapse(__nv_bfloat16* __restrict__ x, float* __restrict__ stats, __nv_bfloat16* __restrict__ u, int T)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int group = tid / 256;
    int gtid = tid - group * 256;
    long long col0 = (long long)(gtid * 8);
    int nsub = T * 2;
    #pragma unroll 1
    for (int row0 = bid; row0 < nsub; row0 += num_bids) {
        int sub = row0 + group;
        if (sub < nsub) {
            int row = sub / 2;
            int part = sub - row * 2;
            long long row64 = (long long)row;
            long long colb = (long long)(part * 2048) + col0;
            long long xbase = row64 * 16384 + colb;
            long long ubase = row64 * 4096 + colb;
            float cache[32];
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                #pragma unroll
                for (int v = 0; v < 1; v++) {
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (xbase + (long long)(s * 4096 + v * 2048)) + 0);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&cache[(s + v) * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                (&cache[(s + v) * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            long long sbase = row64 * 64;
            float _vec_load_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(stats + (sbase + 24) + 0);
                _vec_load_0[0 + 0] = _v4.x;
                _vec_load_0[0 + 1] = _v4.y;
                _vec_load_0[0 + 2] = _v4.z;
                _vec_load_0[0 + 3] = _v4.w;
            }
            float _vec_load_1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(stats + (sbase + 28) + 0);
                _vec_load_1[0 + 0] = _v4.x;
                _vec_load_1[0 + 1] = _v4.y;
                _vec_load_1[0 + 2] = _v4.z;
                _vec_load_1[0 + 3] = _v4.w;
            }
            float pre[4];
            pre[0] = _vec_load_0[1];
            pre[1] = _vec_load_0[2];
            pre[2] = _vec_load_0[3];
            pre[3] = _vec_load_1[0];
            #pragma unroll
            for (int v_1 = 0; v_1 < 1; v_1++) {
                unsigned int uw[4];
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    float _fma_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(pre[1]), "f"(cache[(1 + v_1) * 8 + 2 * k]), "f"(mul_f32_rn_noftz(pre[0], cache[v_1 * 8 + 2 * k])));
                    float _fma_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(pre[2]), "f"(cache[(2 + v_1) * 8 + 2 * k]), "f"(_fma_5));
                    float _fma_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(pre[3]), "f"(cache[(3 + v_1) * 8 + 2 * k]), "f"(_fma_4));
                    float _fma_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(pre[1]), "f"(cache[(1 + v_1) * 8 + 2 * k + 1]), "f"(mul_f32_rn_noftz(pre[0], cache[v_1 * 8 + 2 * k + 1])));
                    float _fma_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(pre[2]), "f"(cache[(2 + v_1) * 8 + 2 * k + 1]), "f"(_fma_8));
                    float _fma_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(pre[3]), "f"(cache[(3 + v_1) * 8 + 2 * k + 1]), "f"(_fma_7));
                    unsigned int hword = __as_u32(__float22bfloat162_rn(make_float2(_fma_3, _fma_6)));
                    uw[k] = hword;
                }
                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(u + (ubase + (long long)(v_1 * 2048))))[0] = reinterpret_cast<int4*>(uw)[0];
            }
        }
    }
}

} // extern "C"
