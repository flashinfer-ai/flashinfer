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
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
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

#include <math_constants.h>

#ifndef CAKE_MEGAMOE_ROUTE_TERM_WORDS_DEFINED
#define CAKE_MEGAMOE_ROUTE_TERM_WORDS_DEFINED
// Loads the eight route-term words of one 8-column group and waits until none of their 32-bit words is all ones,
// i.e. until every expert owner's store has arrived (each word is single-copy atomic; owners store it once per forward).
static __device__ __forceinline__ void cake_megamoe_wait_route_terms(const __nv_bfloat16* route_terms, uint4 (&words)[8]) {
    #pragma unroll
    for (int route_slot = 0; route_slot < 8; route_slot++) {
        asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(words[route_slot].x), "=r"(words[route_slot].y), "=r"(words[route_slot].z), "=r"(words[route_slot].w)
                     : "l"(route_terms + route_slot * 3072) : "memory");
    }
    #pragma unroll
    for (int route_slot = 0; route_slot < 8; route_slot++) {
        while (words[route_slot].x == 0xFFFFFFFFu || words[route_slot].y == 0xFFFFFFFFu ||
               words[route_slot].z == 0xFFFFFFFFu || words[route_slot].w == 0xFFFFFFFFu) {
            __nanosleep(64);
            asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(words[route_slot].x), "=r"(words[route_slot].y), "=r"(words[route_slot].z), "=r"(words[route_slot].w)
                         : "l"(route_terms + route_slot * 3072) : "memory");
        }
    }
}
// Re-arms the eight route-term words of one group for the next forward (peers store them only after its rank barrier 0).
static __device__ __forceinline__ void cake_megamoe_rearm_route_terms(__nv_bfloat16* route_terms) {
    #pragma unroll
    for (int route_slot = 0; route_slot < 8; route_slot++) {
        asm volatile("st.global.v4.b32 [%0], {%1, %1, %1, %1};" :: "l"(route_terms + route_slot * 3072), "r"(0xFFFFFFFFu) : "memory");
    }
}
#endif
extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_mxfp8_megamoe_ep16_2ce7cc644321f53ef127(__nv_bfloat16* __restrict__ route_terms_bf16, __nv_bfloat16* __restrict__ output_bf16, int tokens_per_rank)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int worker = bid * 128 + tid;
    int total_workers = tokens_per_rank * 384;
    if (worker < total_workers) {
        int token = worker / 384;
        int hidden_tile = worker - token * 384;
        int column = hidden_tile * 8;
        unsigned long long route_base = (unsigned long long)token * 8 * 3072 + (unsigned long long)column;
        // Peers store this token's route terms during the fused kernel; wait for all eight words of the group.
        uint4 _route_words[8];
        cake_megamoe_wait_route_terms(route_terms_bf16 + route_base, _route_words);
        cake_megamoe_rearm_route_terms(route_terms_bf16 + route_base);
        float _vec_load_0[8];
        {
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _route_words[0];
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
        float accum[8];
        #pragma unroll
        for (int element = 0; element < 8; element++) {
            accum[element] = _vec_load_0[element];
        }
        #pragma unroll
        for (int route_slot = 1; route_slot < 8; route_slot++) {
            float _vec_load_1[8];
            {
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _route_words[route_slot];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            #pragma unroll
            for (int element_1 = 0; element_1 < 8; element_1++) {
                accum[element_1] = accum[element_1] + _vec_load_1[element_1];
            }
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(accum[0 + 2], accum[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(accum[0 + 4], accum[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(accum[0 + 6], accum[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(output_bf16 + ((unsigned long long)token * 3072 + (unsigned long long)column)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
    }
}

} // extern "C"
