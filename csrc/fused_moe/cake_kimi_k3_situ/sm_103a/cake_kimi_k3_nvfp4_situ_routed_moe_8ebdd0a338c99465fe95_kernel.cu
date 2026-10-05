/*
 * Copyright (c) 2023 by FlashInfer team.
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
#define THREADS 128
#define H 3584

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 4) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_8ebdd0a338c99465fe95(__nv_bfloat16* __restrict__ expert_output, __nv_bfloat16* __restrict__ route_weights, int* __restrict__ token_to_permuted, __nv_bfloat16* __restrict__ out, int M)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int token = blockIdx.x;
    if (token < M) {
        #pragma unroll 1
        for (int feature = tid * 8; feature < H; feature += 1024) {
            float accum[8];
            accum[0] = 0.0f;
            accum[1] = 0.0f;
            accum[2] = 0.0f;
            accum[3] = 0.0f;
            accum[4] = 0.0f;
            accum[5] = 0.0f;
            accum[6] = 0.0f;
            accum[7] = 0.0f;
            #pragma unroll
            for (int slot = 0; slot < 16; slot++) {
                int pair = token * 16 + slot;
                int row = token_to_permuted[pair];
                float weight = route_weights[pair];
                float _vec_load_0[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(expert_output + (unsigned int)(row * H + feature) + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int elem = 0; elem < 8; elem++) {
                    accum[elem] = accum[elem] + weight * _vec_load_0[elem];
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(accum[0 + 2], accum[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(accum[0 + 4], accum[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(accum[0 + 6], accum[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (token * H + feature)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"
